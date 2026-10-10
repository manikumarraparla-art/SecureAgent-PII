"""Document processing inside the PII boundary; mappings never persist.

Document tokens are irreversible and cannot alias user tokens. At retrieval
all token-shaped strings (including injected ones) move to DOC namespace.
"""
import hashlib
from app.security.gateway import normalize_input
from app.security.tokenizer import TOKEN_PATTERN, TokenVault


def protect_document(text, gateway, document_id, request_id=None):
    # Escape hostile placeholders before allocating real document tokens.
    text = TOKEN_PATTERN.sub(lambda m: "[document placeholder " + m.group(1) + "]", normalize_input(text))
    vault = TokenVault()
    try:
        protected = gateway.protect(text, request_id=request_id, vault=vault)
        gateway.verify_no_leak(protected.sanitized_text, vault)
        def namespace(match):
            identity = document_id + match.group(0)
            number = int(hashlib.sha256(identity.encode()).hexdigest(), 16)
            return f"<DOC_{match.group(1)}_{number}>"
        sanitized = TOKEN_PATTERN.sub(namespace, protected.sanitized_text)
        gateway.verify_no_leak(sanitized, vault)
        return sanitized, protected.entity_counts
    finally:
        vault.clear()


def protect_retrieved(text, gateway, document_id, request_id, vault):
    # Keep persisted placeholders distinct from any new tokens. They carry
    # no reversible values; even a tampered index cannot alias the user vault.
    tokens = {}
    def hold(match):
        marker = f"[stored document placeholder {len(tokens)}]"
        number = int(hashlib.sha256((document_id + match.group(0)).encode()).hexdigest(),16)
        tokens[marker] = f"<DOC_{match.group(1).removeprefix('DOC_')}_{number}>"
        return marker
    held = TOKEN_PATTERN.sub(hold, normalize_input(text))
    protected = gateway.protect(held, request_id=request_id, vault=vault)
    gateway.verify_no_leak(protected.sanitized_text, vault)
    def namespace(match):
        number = int(hashlib.sha256((document_id + match.group(0)).encode()).hexdigest(),16)
        return f"<DOC_{match.group(1)}_{number}>"
    sanitized = TOKEN_PATTERN.sub(namespace, protected.sanitized_text)
    counts = protected.entity_counts
    for marker, token in tokens.items():
        sanitized = sanitized.replace(marker, token)
    return sanitized, counts
