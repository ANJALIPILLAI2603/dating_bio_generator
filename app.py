from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re
import os

app = Flask(__name__)

# Initialize model and tokenizer explicitly for better control
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium", 
    truncation_side='left',  # or 'right'
    truncation=True,  # Explicitly enable truncation
    model_max_length=512  # Set an appropriate max length
)

# Separate model initialization
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

# Create pipeline with the initialized model and tokenizer
generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    framework="pt",
    device=-1  # Use CPU. Change to 0 for GPU if available
)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def generate_bio():
    try:
        data = request.get_json()
        
        profession = data.get('profession', '').strip()
        interests = data.get('interests', '').strip()
        hobbies = data.get('hobbies', '').strip()
        goals = data.get('goals', '').strip()
        
        if not all([profession, interests, hobbies, goals]):
            return jsonify({"error": "All fields are required"}), 400

        # Create multiple prompt variations for diversity
        prompts = [
            f"Dating Profile: {profession} seeking {goals}. My passion for {interests} drives me, and I love spending time {hobbies}. I believe that",
            
            f"Bio: Tech professional ({profession}) with a creative side. When I'm not working, I'm immersed in {interests} or {hobbies}. Looking for {goals} with someone who",
            
            f"About me: Balancing life as a {profession} while pursuing my interests in {interests}. You'll often find me {hobbies}. Seeking {goals} because"
        ]
        
        generated_bios = []
        for prompt in prompts:
            # Generate with different parameters for variety
            response = generator(
    prompt,
    max_length=150,
    num_return_sequences=2,
    temperature=0.9,
    top_p=0.92,
    do_sample=True,
    no_repeat_ngram_size=2,
    top_k=50,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

            for gen_text in response:
                cleaned_bio = clean_bio(gen_text['generated_text'], prompt)
                if is_valid_bio(cleaned_bio, profession, interests, hobbies):
                    generated_bios.append(cleaned_bio)

        # If we got any valid generations, choose the best one
        if generated_bios:
            best_bio = choose_best_bio(generated_bios, profession, interests, hobbies)
            return jsonify({"bio": best_bio})
        
        # If all generations failed, try one more time with different parameters
        backup_prompt = (
            f"Profile of a {profession}: Beyond my work in technology, I'm deeply passionate about "
            f"{interests}. In my free time, you'll find me {hobbies}. "
            f"I'm here because I'm looking for {goals} with someone who"
        )
        
        backup_response = generator(
    backup_prompt,
    max_length=200,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
        
        backup_bio = clean_bio(backup_response[0]['generated_text'], backup_prompt)
        
        # Only use fallback if backup generation also fails
        if not is_valid_bio(backup_bio, profession, interests, hobbies):
            backup_bio = create_dynamic_fallback_bio(profession, interests, hobbies, goals)

        return jsonify({"bio": backup_bio})

    except Exception as e:
        print(f"Error generating bio: {str(e)}")
        return jsonify({"error": "Failed to generate bio"}), 500

def clean_bio(text, prompt):
    """Clean and format the generated bio text."""
    # Remove the prompt
    text = text.replace(prompt, '')
    
    # Remove common prefixes
    prefixes = ['Dating Profile:', 'Bio:', 'About me:', 'Profile:']
    for prefix in prefixes:
        text = re.sub(f'^{prefix}', '', text, flags=re.IGNORECASE).strip()
    
    # Clean up whitespace
    text = ' '.join(text.split())
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    complete_sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    
    if not complete_sentences:
        return text
    
    # Take up to 4 complete sentences
    final_text = '. '.join(complete_sentences[:4]).strip()
    
    # Ensure proper capitalization and punctuation
    if final_text:
        final_text = final_text[0].upper() + final_text[1:]
        if not final_text[-1] in '.!?':
            final_text += '.'
    
    return final_text

def is_valid_bio(bio, profession, interests, hobbies):
    """Check if the bio meets our quality criteria."""
    if not bio or len(bio.split()) < 20:
        return False
        
    # Check for key content inclusion
    keywords = [profession.lower(), interests.lower(), hobbies.lower()]
    word_count = sum(1 for keyword in keywords if keyword in bio.lower())
    
    # At least 2 of the 3 key elements should be present
    return word_count >= 2

def choose_best_bio(bios, profession, interests, hobbies):
    """Select the best bio from the generated options."""
    scored_bios = []
    for bio in bios:
        score = 0
        # Length score (prefer longer, but not too long)
        words = len(bio.split())
        if 30 <= words <= 100:
            score += 5
        
        # Keyword inclusion score
        keywords = [profession.lower(), interests.lower(), hobbies.lower()]
        score += sum(3 for keyword in keywords if keyword in bio.lower())
        
        # Sentence structure score
        sentences = re.split(r'[.!?]+', bio)
        if 2 <= len(sentences) <= 4:
            score += 3
            
        scored_bios.append((score, bio))
    
    return max(scored_bios, key=lambda x: x[0])[1]

def create_dynamic_fallback_bio(profession, interests, hobbies, goals):
    """Create a more varied fallback bio using templates."""
    templates = [
        f"As a {profession}, I blend  expertise with my passion for {interests}. You'll often find me {hobbies} when I'm not crafting elegant solutions. Seeking {goals} with someone who appreciates both innovation and adventure.",
        
        f"Creative {profession} with a love for {interests} and {hobbies}. My work keeps me analytical, while my hobbies keep me grounded. Looking for {goals} with someone who values growth and genuine connections.",
        
        f"Tech-savvy {profession} by day, enthusiastic {interests} enthusiast by night. {hobbies} helps me maintain work-life balance. Hoping to find {goals} with someone who shares my curiosity about life."
    ]
    
    import random
    return random.choice(templates)



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)