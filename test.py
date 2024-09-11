import google.generativeai as palm
import key_param
palm.configure(api_key=key_param.google_api_key)

models = [m for m in palm.list_models() if 'embedText' in m.supported_generation_methods]
if models:
    print(f"Available embedding models: {models}")
else:
    print("No embedding models available. Check your API key and permissions.")