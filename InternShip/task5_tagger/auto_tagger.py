# task5_tagger/auto_tagger.py
import time
from transformers import pipeline

print("🚀 Step 1: Initializing Zero-Shot NLP Tagging Model (BART)...")
print("*(Note: This will download a lightweight pre-trained model on its first run. Please wait...)*")

# Using the industry standard facebook/bart-large-mnli for zero-shot text classification
try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Retrying with a smaller model...")
    classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")

# 2. Define a free-text support ticket dataset
tickets = [
    "My screen goes completely black when trying to open the app settings.",
    "I've been charged twice for my subscription this month. Please refund the $29.99.",
    "How can I change the registered email address on my enterprise dashboard?",
    "Your server is returning a 502 bad gateway error on the billing page."
]

# 3. Define candidate categories (Tags)
candidate_labels = ["Billing & Finance", "Technical Support", "Account Management", "General Inquiry"]

print("\n==================================================")
print("       ZERO-SHOT CLASSIFICATION RUN (TOP 3 TAGS)")
print("==================================================")

# Loop through each ticket and output top 3 probable tags
for i, ticket in enumerate(tickets, 1):
    start_time = time.time()
    res = classifier(ticket, candidate_labels)
    elapsed = time.time() - start_time
    
    print(f"\n📨 [Ticket #{i}]: '{ticket}'")
    print(f"⏱️ Prediction Time: {elapsed:.2f} seconds")
    print("🏷️ Top 3 Tag Rankings:")
    
    # Extract and display top 3 labels and scores
    for rank, (label, score) in enumerate(zip(res['labels'][:3], res['scores'][:3]), 1):
        print(f"   {rank}. {label:<20} | Probability: {score:.2%}")
    print("-" * 50)

print("\n==================================================")
print("           FEW-SHOT PROMPTING TEMPLATE")
print("==================================================")
# Few-shot learning relies on passing structured examples to show an LLM how to output classifications.
# Here we print the exact prompt payload structured for a Large Language Model.

few_shot_prompt_template = """
You are an AI Customer Support Agent. Tag the user's ticket into one of the following classes: 
[Billing & Finance, Technical Support, Account Management, General Inquiry]. Output only the tag.

Example 1:
Ticket: "I forgot my password and cannot log into my workspace."
Tag: Account Management

Example 2:
Ticket: "Can I get a PDF copy of my last invoice?"
Tag: Billing & Finance

Example 3:
Ticket: "The database connection keeps timing out after 30 seconds."
Tag: Technical Support

Your Turn:
Ticket: "{user_ticket}"
Tag:"""

# Print how a few-shot pipeline structures the prompt for the model
print("Below is how the few-shot context is constructed and fed to the LLM:")
sample_prompt = few_shot_prompt_template.format(user_ticket=tickets[0])
print(sample_prompt)