"""Request-local tool/document vaults; only sanitized data leaves this layer."""
from dataclasses import dataclass, field
import json
import uuid
from app.security.gateway import normalize_input
from app.security.tokenizer import TOKEN_PATTERN, TokenVault

@dataclass
class ToolSecuritySession:
    request_id: str
    gateway: object = field(repr=False)
    max_result_chars: int
    vault: TokenVault = field(default_factory=TokenVault,repr=False)
    sources: list = field(default_factory=list,repr=False)
    document_vaults: dict = field(default_factory=dict,repr=False)
    entity_counts: dict = field(default_factory=dict)
    document_entity_counts: dict = field(default_factory=dict)
    tool_pii_detected: bool = False
    document_pii_detected: bool = False

    def sanitize(self, value, preserve_documents=False):
        text=json.dumps(value,ensure_ascii=False,allow_nan=False)
        if len(text)>self.max_result_chars:
            from app.agent.errors import ToolResultTooLarge
            raise ToolResultTooLarge("Tool result too large")
        # Tokenize each key and leaf separately: a detector span over serialized
        # JSON can cross quotes and separators, and replacing it corrupts the
        # structure (e.g. a phone match spanning '-0", "score": 0.57...').
        keys={}  # keys repeat across list items; values are scanned per occurrence
        def key(item):
            if item not in keys: keys[item]=self._sanitize_text(item,preserve_documents)
            return keys[item]
        def leaf(item): return self._sanitize_text(item,preserve_documents)
        def walk(item,depth=0):
            if depth>32: raise ValueError("Tool result nesting too deep")
            if isinstance(item,dict): return {key(str(k)):walk(v,depth+1) for k,v in item.items()}
            if isinstance(item,list): return [walk(v,depth+1) for v in item]
            if isinstance(item,str): return leaf(item)
            if item is None or isinstance(item,bool): return item
            # Integral numbers can carry phone/SSN digits; a fractional decimal
            # (e.g. a similarity score) cannot represent a supported PII type.
            if isinstance(item,float) and not item.is_integer(): return item
            safe=leaf(json.dumps(item))
            return item if safe==json.dumps(item) else safe
        result=walk(json.loads(text))
        safe=json.dumps(result,ensure_ascii=False)
        # Normalization and JSON escaping may otherwise hide known PII on wire.
        self.verify(safe)
        if len(safe)>self.max_result_chars:
            from app.agent.errors import ToolResultTooLarge
            raise ToolResultTooLarge("Sanitized result too large")
        return result

    def _sanitize_text(self, text, preserve_documents):
        text=normalize_input(text)
        held={}
        nonce=uuid.uuid4().hex
        def hold(match):
            if preserve_documents and match.group(1).startswith('DOC_'):
                marker=f"[safe document {nonce} {len(held)}]"
                held[marker]=match.group(0)
                self.document_pii_detected=True
                kind={"DOC_PERSON":"PERSON","DOC_EMAIL":"EMAIL_ADDRESS",
                      "DOC_PHONE":"PHONE_NUMBER","DOC_SSN":"US_SSN"}.get(match.group(1))
                if kind:
                    self.document_entity_counts[kind]=self.document_entity_counts.get(kind,0)+1
                return marker
            # Foreign token lookalikes must not alias user or genuine tool values.
            return "[literal placeholder " + match.group(1) + "]"
        text=TOKEN_PATTERN.sub(hold,text)
        protected=self.gateway.protect(text,request_id=self.request_id,vault=self.vault)
        self.gateway.verify_no_leak(protected.sanitized_text,self.vault)
        for kind,count in protected.entity_counts.items():
            self.entity_counts[kind]=self.entity_counts.get(kind,0)+count
        self.tool_pii_detected |= bool(protected.total_entities)
        safe=TOKEN_PATTERN.sub(lambda m:f"<TOOL_{m.group(1)}_{m.group(2)}>",protected.sanitized_text)
        for marker,token in held.items(): safe=safe.replace(marker,token)
        self.verify(safe)
        return safe

    def verify(self, text):
        self.gateway.verify_no_leak(normalize_input(text),self.vault)
        for vault in self.document_vaults.values():
            self.gateway.verify_no_leak(normalize_input(text),vault)

    @property
    def pii_counts(self):
        counts=dict(self.entity_counts)
        for kind,count in self.document_entity_counts.items():
            counts[kind]=counts.get(kind,0)+count
        return counts

    def clear(self):
        self.vault.clear()
        for vault in self.document_vaults.values(): vault.clear()
        self.document_vaults.clear()
        self.sources.clear()
