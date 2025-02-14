from langchain_community.llms import Ollama
from retriever import course_keywords
import re
from retriever import count_candidates_from_university

llm = Ollama(model="deepseek-r1:8b")

def clean_response(response_text):
    # Remove any reasoning within <think> tags
    response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
    return response_text.strip()

def generate_llm_response(mapped_results, question):
    question_lower = question.lower()

    # 🔹 Detect "How many candidates from [University]?"
    if "how many" in question_lower and ("university" in question_lower or "universiti" in question_lower):
        university_name = question_lower.replace("how many candidates from", "").strip()
        count = count_candidates_from_university(university_name)

        if isinstance(count, int):
            if count > 0:
                return f"There are **{count}** candidates from **{university_name}**."
            else:
                return f"No candidates found from **{university_name}**."
        else:
            return count  # Return error message if CSV loading failed


     # Detect "how many" queries
    if "how many" in question_lower:
        # Extract course name from query
        course_name = next((keyword for keyword in course_keywords if keyword in question_lower), None)
        
        if course_name:
            # Count unique candidates pursuing the specified course
            unique_candidates = {res["original_file"] for res in mapped_results if res["course"].lower() == course_name.lower()}
            count = len(unique_candidates)

            if count > 0:
                return f"There are **{count}** candidates pursuing **{course_name}**."
            else:
                return f"Sources list possible candidates pursuing **{course_name}**."
        else:
            return f"Sources list possible candidates pursuing **{course_name}**."

    # Greeting & Farewell Handling
    greetings = ["hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    farewells = ["bye", "goodbye", "see you", "take care", "good night"]

    # Keywords to detect relevant queries
    relevant_keywords = ["ai", "candidate", "experience", "skills", "education", "project", "role", "intern", "give",
                        "finance", "computer science", "hr", "it", "accounting", "business", "cgpa", "applicant", "anyone", "info", "human resource", "management", "who", "summarize", "summarise"]

    # Check for relevant keywords first
    if not any(keyword in question_lower for keyword in relevant_keywords):
        if any(greeting in question_lower for greeting in greetings):
            return "Hello! How can I assist you with candidate matching today?"
        
        if any(farewell in question_lower for farewell in farewells):
            return "Goodbye! Feel free to return if you need more assistance. Have a great day!"
        
        return "I'm here to help with candidate matching. Please ask about the candidates, their skills, experience, or roles."

    # 🔹 Step 1: Check if the query is a name search
    words = question.lower().split()
    name_match = next((word for word in words if len(word) > 2 and word[0].isupper()), None)

    if name_match:
        # 🔹 Filter mapped_results to include only candidates whose names match
        filtered_candidates = [res for res in mapped_results if name_match.lower() in res["text"].lower()]
        
        if not filtered_candidates:
            return f"No candidate found with the name **{name_match}**."
        
        sources = "\n".join([f"- {res['original_file']}" for res in filtered_candidates])
        context = "\n\n".join([res["text"] for res in filtered_candidates])
    
    else:
        # 🔹 If it's a general query, use all results
        sources = "\n".join([f"- {res['original_file']}" for res in mapped_results])
        context = "\n\n".join([res["text"] for res in mapped_results])

    # 🔹 Step 2: Generate Response with LLM
    # Group results by Resume ID to prevent data mixing
    grouped_candidates = {}
    for res in mapped_results:
        resume_id = res["original_file"]
        if resume_id not in grouped_candidates:
            grouped_candidates[resume_id] = []
        grouped_candidates[resume_id].append(res)

    # 🔹 Step 4: If still no candidates, return a response immediately
    if not grouped_candidates:
        return "I'm sorry, I couldn't find any candidates matching your request."

    # Generate response per candidate
    response_text = ""
    for resume_id, candidate_chunks in grouped_candidates.items():
        sources = "\n".join([f"- {chunk['original_file']}" for chunk in candidate_chunks])
        context = "\n\n".join([chunk["text"] for chunk in candidate_chunks])
        
        prompt = f"""
        You are an AI assistant helping HR managers find the best candidates. Answer concisely using bullet points.
        Keep each candidate's details separate and structured.
        Candidate name should be header bold.
        Answer concisely in bullet points, and **ONLY** use retrieved resume data.
        **STRICT RULES:**
        - If you cannot find relevant information, respond with: "I'm sorry, I don't have that information."
        - DO NOT guess, assume, or create information beyond the provided text.
        - DO NOT include any candidate unless they are found in the provided sources.

        
        **Query:** '{question}'

        **Example Response Format:**
        - **Candidate Name:** [Extracted from text]
        - **University:** [Extracted if available]
        - **Course:** [Extracted if available]
        - **Experience:** [Extracted if available]
        - **Education:** [Extracted if available]
        - **Result:** (CGPA/GPA) [If available]
        - **Skills:** (Key relevant skills only)
        - **Projects:** [Mention briefly if applicable]
        
        **Sources:**
        {sources}

        **Context:**
        {context}
        """
        raw_response = llm.invoke(prompt)
        response_text += clean_response(raw_response) + "\n\n"
    
    return response_text.strip()