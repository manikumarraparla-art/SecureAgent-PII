"""Run locally: python -m streamlit run app/ui/app.py."""
import os
import streamlit as st
from dotenv import load_dotenv
from secureagent_ui.api_client import APIClient, UIFailure, FailureCode, MESSAGES, PendingApproval
from secureagent_ui.contracts import Mode, ROLES, RESTORABLE_TYPES
from secureagent_ui.state import Conversation
from secureagent_ui.components import show_response, show_pending

DEMO_PROMPTS={
    Mode.CHAT:"My name is David Chen and my email is david.chen@example.com. Please acknowledge my request.",
    Mode.RAG:"What is the status of claim CLM-1001?",
    Mode.AGENT:"What is the status of claim CLM-1001?",
}

STYLE="""<style>
.stApp { background: #f6f8fb; }
.block-container { max-width: 1120px; padding-top: 2.3rem; padding-bottom: 3rem; }
h1 { letter-spacing: -0.045em; }
[data-testid="stSidebar"] { background: #eaf0f5; border-right: 1px solid #dce4ed; }
[data-testid="stChatMessage"] { background: #fff; border: 1px solid #e1e7ef; border-radius: 12px; }
[data-testid="stText"] { font-family: inherit; line-height: 1.65; white-space: pre-wrap; overflow-wrap: anywhere; }
[data-testid="stMetricValue"] { font-size: 1.3rem; }
</style>"""

def clear_conversation():
    conversation=st.session_state.get('conversation')
    if conversation is not None: conversation.clear()
    for key in ('draft','message_input','sample_text'): st.session_state.pop(key,None)


def _record(conversation,received,mode):
    if isinstance(received,PendingApproval): conversation.add_pending(received,mode)
    else: conversation.add_response(received,mode)


def main():
    load_dotenv()
    st.set_page_config(page_title='SecureAgent-PII',page_icon='▣',layout='wide',initial_sidebar_state='expanded')
    st.markdown(STYLE,unsafe_allow_html=True)  # Fixed local CSS only; never user/model HTML.
    st.title('SecureAgent-PII')
    st.caption('Secure Local Enterprise AI Assistant')
    if 'conversation' not in st.session_state: st.session_state['conversation']=Conversation()
    conversation=st.session_state['conversation']
    with st.sidebar:
        st.subheader('Workspace')
        mode=Mode(st.selectbox('Operating mode',[m.value for m in Mode],key='mode'))
        role='customer_service'
        if mode is Mode.AGENT:
            role=st.selectbox('Demo Role',ROLES,key='demo_role')
            st.caption('Demo roles are used for local authorization testing. Production identity would be supplied by an authenticated identity provider.')
        top_k=st.slider('Sources to retrieve',1,10,3,key='top_k') if mode is Mode.RAG else 3
        allowed=st.multiselect('Restore user PII types',RESTORABLE_TYPES,key='restore_types',
            help='Default: none. The backend applies its server policy. Tool and document values remain protected.')
        st.button('Clear conversation',on_click=clear_conversation,key='clear',width='stretch')
        st.caption('Session memory only · Requests are independent')
        with st.expander('Synthetic Demo Data'):
            st.text(DEMO_PROMPTS[mode])
            if st.button('Use sample prompt',key='sample',width='stretch'):
                st.session_state['draft']=DEMO_PROMPTS[mode]
            if mode is Mode.AGENT:
                st.caption('Authorization demo: customer_service + CLM-1001 should receive a safe tool-access denial.')
                st.caption('Approval demo: claims_adjuster + CLM-1004 pauses for a supervisor decision.')
        with st.expander('Security Demonstration'):
            st.caption('Detected PII is tokenized before model processing. No private mapping is sent to this UI.')
            for kind in ('PERSON','EMAIL_ADDRESS','US_SSN'): st.text(f'{kind} → protected')
    try:
        connect=float(os.environ.get('UI_CONNECT_TIMEOUT','5'))
        timeout=float(os.environ.get('UI_RESPONSE_TIMEOUT','180'))
        client=APIClient(os.environ.get('UI_BACKEND_URL','http://127.0.0.1:8000'),connect,timeout)
    except Exception:
        st.error(MESSAGES[FailureCode.CONFIGURATION]);return
    with client:
        health=client.health()
        a,b,c,d=st.columns([1,1,1,1.3])
        a.metric('Backend','Available' if health.backend_available else 'Unavailable')
        b.metric('Provider','Foundry Local' if health.provider == 'foundry' else 'Ollama')
        c.metric('LLM','Available' if health.llm_available else 'Unavailable')
        d.metric('Model',health.model)
        st.caption(f'Backend version: {health.version}')
        if mode is Mode.AGENT and not health.tool_calling:
            st.warning(MESSAGES[FailureCode.CAPABILITY_UNSUPPORTED])
        if not health.backend_available: st.warning(MESSAGES[FailureCode.BACKEND_UNAVAILABLE])
        elif not health.llm_available: st.warning(MESSAGES[FailureCode.LLM_UNAVAILABLE])
        if st.button('Refresh status',key='refresh'): st.rerun()
        st.divider()
        if not conversation.messages:
            with st.container(border=True):
                st.subheader('A secure workspace for enterprise questions')
                st.write('Choose a mode, ask a question, and review the security status alongside each answer.')
                st.caption('Use the fictional claim and policy IDs in Synthetic Demo Data. No conversation is stored on disk.')
        decision=None
        for index,item in enumerate(conversation.messages):
            with st.chat_message(item.speaker):
                st.caption(item.mode.value)
                if item.received is not None: show_response(item.received)
                elif item.error is not None: st.error(item.text)
                elif item.pending is not None:
                    show_pending(item.pending)
                    approval_id=item.pending.approval.approval.approval_id
                    if not conversation.is_decided(approval_id):
                        left,right=st.columns(2)
                        if left.button('Approve as supervisor (demo)',key=f'approve-{index}'): decision=(approval_id,'approve')
                        if right.button('Deny',key=f'deny-{index}'): decision=(approval_id,'deny')
                else: st.text(item.text)
        if decision:
            conversation.mark_decided(decision[0])
            with st.spinner('Resuming the paused graph…'):
                try: _record(conversation,client.decide(*decision),Mode.AGENT)
                except UIFailure as exc: conversation.add_error(exc.code,Mode.AGENT)
                except Exception: conversation.add_error(FailureCode.SERVER_ERROR,Mode.AGENT)
            st.rerun()
        draft=st.session_state.get('draft')
        if draft:
            with st.form('sample_composer',clear_on_submit=True):
                text=st.text_area('Synthetic sample — edit before sending',value=draft,max_chars=8000,key='sample_text')
                send=st.form_submit_button('Send sample',disabled=not health.backend_available)
            if send:
                st.session_state.pop('draft',None)
                prompt=text
            else: prompt=None
        else:
            prompt=st.chat_input('Ask an enterprise question…',max_chars=8000,key='message_input',disabled=not health.backend_available)
        if prompt:
            conversation.add_user(prompt,mode)
            with st.spinner('Processing through the secure backend…'):
                try: _record(conversation,client.send(mode,prompt,role,allowed,top_k),mode)
                except UIFailure as exc: conversation.add_error(exc.code,mode)
                except Exception: conversation.add_error(FailureCode.SERVER_ERROR,mode)
            st.rerun()

if __name__=='__main__': main()
