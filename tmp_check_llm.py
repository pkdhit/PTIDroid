from llm_client import load_llm_settings, make_openai_client, extract_openai_chat_content
import json
s=load_llm_settings('BASELINE')
c=make_openai_client(s.base_url, s.api_key)
print('client_none', c is None)
print('has_chat', hasattr(c, 'chat'))
print('has_responses', hasattr(c, 'responses'))
r=c.chat.completions.create(model=s.model, messages=[{'role': 'user', 'content': 'Reply only with JSON: {"ok":true}'}], temperature=0.2, max_tokens=50, timeout=60)
print('type', type(r).__name__)
print('content', repr(extract_openai_chat_content(r)))
print('has_model_dump', hasattr(r, 'model_dump'))
try:
    d=r.model_dump()
    print('dump_keys', list(d.keys()))
    print('dump', json.dumps(d, ensure_ascii=False, default=str, indent=2))
except Exception as e:
    print('dump_err', repr(e))
