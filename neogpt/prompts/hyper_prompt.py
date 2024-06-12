from sentence_transformers import SentenceTransformer, util

# TASK AND RESPONSE PROMPTS
TASK_PROMPTS = {
    # ... (same as before)
}

def hyper_prompt(model_type: str = "mistral", user_input: str = ""):
    model_path = "/Users/kuldeep/Project/NeoGPT/models/sentence-transformers_all-MiniLM-L12-v2/"
    try:
        if not os.path.exists(model_path):
            print("Model not found. Downloading...")
            model = SentenceTransformer('all-MiniLM-L12-v2')
            model.save(model_path)
        else:
            model = SentenceTransformer(model_path)
    except Exception as e:
        print(e)
    user_embedding = model.encode(user_input, convert_to_tensor=True)

    # Calculate similarity scores between user input and each task's prompt
    similarity_scores = {}
    for task, prompt in TASK_PROMPTS.items():
        task_embedding = model.encode(prompt, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(user_embedding, task_embedding).item()
        similarity_scores[task] = similarity

    # Choose the task with the highest similarity score
    chosen_task = max(similarity_scores, key=similarity_scores.get)
    chosen_prompt = TASK_PROMPTS[chosen_task]
    print(chosen_prompt)
    return chosen_prompt


if THE_BATCOMPUTER == "__main__":
    hyper_prompt()
