"""Plain-text rendering of validated public metadata. No raw internal objects."""
import streamlit as st
from secureagent_ui.api_client import ReceivedResponse

def security_rows(metadata):
    return [('PII detected','Yes' if metadata.pii_detected else 'No'),
            ('Detected types',', '.join(metadata.entity_types) or 'None detected'),
            ('Document PII detected','Yes' if metadata.document_pii_detected else 'No'),
            ('Tool PII detected','Yes' if metadata.tool_pii_detected else 'No'),
            ('Re-identification',metadata.reidentification_decision)]

def source_rows(sources):
    return [{'Filename':s.filename,'Document ID':s.document_id,'Chunk ID':s.chunk_id,
             'Similarity':f'{s.score:.3f}'} for s in sources]

def tool_rows(result):
    return [('Tools used',', '.join(result.tools_used) or 'None'),
            ('Authorization','Allowed' if result.tools_used else 'No tool requested'),
            ('Agent steps',str(result.agent_steps) if result.agent_steps is not None else 'Not applicable')]

ROUTES={'chat':'Secure Chat','rag':'Secure RAG','agent':'Secure Agent'}
APPROVALS={'NOT_REQUIRED':'Not required','GRANTED':'Granted (demo supervisor)','DENIED':'Denied'}

def orchestration_rows(orchestration):
    """Safe execution metadata: route and node names, never state or reasoning."""
    rows=[('Orchestration','LangGraph'),('Route',ROUTES[orchestration.route]),
          ('Graph path',' → '.join(orchestration.nodes))]
    if orchestration.tool_protocol:
        rows+=[('Tool protocol',f'MCP ({orchestration.mcp_transport})'),
               ('MCP servers',', '.join(orchestration.mcp_servers) or 'None used'),
               ('Human approval',APPROVALS.get(orchestration.approval,'Not required'))]
    return rows

def approval_rows(pending):
    approval=pending.approval.approval
    return [('Tool',approval.tool),('Record',approval.record_id or 'Not applicable'),
            ('Reason','Restricted synthetic record'),('Approver','supervisor (demo role)'),
            ('Expires in',f'{approval.expires_in_seconds} seconds')]

def show_pending(pending):
    st.warning('This tool action needs supervisor approval. The LangGraph run is paused before the MCP call.')
    for label,value in approval_rows(pending): st.text(f'{label}: {value}')
    st.caption('Demonstration approval only: demo roles are not authenticated identities.')

def show_response(received:ReceivedResponse):
    result=received.result
    st.text(result.response)
    st.caption(f'{result.model} · Round trip {received.latency_ms/1000:.2f}s · Request {result.request_id}')
    with st.expander('Security status'):
        for label,value in security_rows(result.security):
            st.text(f'{label}: {value}')
    if result.sources:
        with st.expander('Sources',expanded=True):
            for source in source_rows(result.sources):
                st.text(source['Filename'])
                st.caption(f"Document {source['Document ID']} · Chunk {source['Chunk ID']} · Similarity {source['Similarity']}")
    if result.agent_steps is not None:
        with st.expander('Tool activity',expanded=True):
            for label,value in tool_rows(result): st.text(f'{label}: {value}')
    if result.orchestration is not None:
        with st.expander('Orchestration'):
            for label,value in orchestration_rows(result.orchestration): st.text(f'{label}: {value}')
