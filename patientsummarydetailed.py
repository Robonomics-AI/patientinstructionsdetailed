import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from time import time


class AzureOpenAIWrapper:
    """
    A wrapper class for Azure OpenAI interaction, promoting code readability
    and maintainability.
    """

    def __init__(self):
        """
        Loads environment variables for API configuration securely.
        """
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        self.api_version = os.getenv("API_VERSION")
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.azure_model_deployment = os.getenv("AZURE_MODEL_DEPLOYMENT")

    def create_chat_completion(self, prompt, max_tokens=3000, temperature=0.2, top_p=0.95):
        """
        Performs chat completion using the configured Azure OpenAI model.

        Args:
            prompt (str): The prompt text for the conversation construction.
            max_tokens (int, optional): The maximum number of tokens allowed in the response. Defaults to 15000.
            temperature (float, optional): Controls the randomness of the generated text. Defaults to 0.2.
            top_p (float, optional): The probability of picking the top words in the vocabulary. Defaults to 0.95.

        Returns:
            str: The generated conversation reconstruction.
        """

        client = AzureOpenAI(api_key=self.api_key,
                             api_version=self.api_version,
                             azure_endpoint=self.azure_endpoint)
        start_time = time()
        response = client.chat.completions.create(
            model=self.azure_model_deployment,  # Replace with the appropriate Azure OpenAI engine
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=[{"role": "system", "content": "Assistant is a conversation constructor between the doctor and "
                                                    "patient."},
                      {"role": "user", "content": f"{prompt}"}]
        )

        total_time = time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"The time difference is: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
        print(f"The number of tokens being used are {response.usage.total_tokens}")
        return response.choices[0].message.content


def summarize_text(json_transcription):
    text = json_transcription["conversation"]
    language = json_transcription["language"]
    medicalliteracylevel = json_transcription["medicalliteracylevel"]
    agegroup = json_transcription["agegroup"]
    length = json_transcription["length"]

    prompt = (
        f"""
         You are an AI assistant of a doctor. Your task is to understand the spoken conversation {text} between the doctor, patient, and patient's attendants (if present) during a consultation, which could be either in person or virtual. Based on the conversation and other available patient histories, prepare a detailed “instruction cum information book” to educate the patient and their caregivers about the patient's condition and explain in detail what needs to be done to adhere to the treatment plan. 

 

The document needs to be in {length} manner in {language} language and should be easy for the patient, 
who is in the age group of {agegroup} and has a medical literacy level of {medicalliteracylevel} 
to understand and follow. 

The document should include the following sections, arranged in a logical, engaging flow: 

Patient Condition:Describe the patient's condition in simple {language}, avoiding overly technical terms. 

Causes:Explain the potential causes of the condition clearly and concisely. Explain the disease causes using appropriate
 analogies. 

Symptoms: List the common symptoms associated with the condition and any specific symptoms identified during the 
consultation. 

Treatment Plan: Explain the prescribed treatment plan, including the rationale behind medications or other 
interventions. Use analogies where appropriate. 

Self-Care: Provide detailed instructions on how the patient and caregiver can manage the condition at home, 
including medication administration, symptom monitoring, and self-care practices. 

Medications: Make a schedule/timetable of when the patient needs to take the prescribed medications, dosage, 
frequency etc. 

Reminders: Include any reminders for the patient, such as potential side effects of medications or signs/symptoms 
to watch out for. 

Tests and Procedures: In bullet points, provide the tests and procedures recommended by the doctor with preparation 
instructions for each test/procedure (e.g., fasting, medication restrictions) 

Activities and Exercises: In bullet points, provide the activities and exercises recommended by the doctor with the 
frequency and duration of the exercises, if any. Also, provide restrictions on the exercises and any 
lifestyle modifications. 

Diet and Nutrition: If applicable, list out bullet points of diet and nutrition recommendations by the 
doctor with specific food or drink restrictions 

Wound Care (if applicable): In bullet points, provide the wound care instructions and signs and symptoms of infection. 

Follow-up Appointments: Schedule of upcoming appointments with healthcare professionals. Explain the importance of 
follow-up appointments and what to expect during those visits. 

Summary: A crisp summary of the consultation for the patient to understand the conversation and to understand the 
reason for the consultation and the treatment plan. 

Further Reading: Include a list of reputable resources (websites, patient support groups) where the patient and 
caregiver can find additional information about the condition. 

Glossary: Including a glossary of medical terms generally associated with the kind os symptoms, and 
treatment plans suggested for the patient. 

End with a motivational quote, encouraging a patient to look after themselves. 

If available, provide the contact information for the healthcare provider or emergency services in case of any 
urgent concerns. 


Other Instructions: 

Ensure that the text generated is grounded on evidence-based research. Do not assume or make up information 
not available. 

Maintain patient confidentiality by avoiding any personal details beyond what's necessary for medical documentation. 

Consider using visuals like diagrams or illustrations to enhance understanding (optional). 

Use a tone that is empathetic, supportive, informative, respectful and easy to understand for patients and caregivers. 

Medical terminology should be explained in simpler terms or avoided when possible. 

You should not claim to be a licensed medical professional and disclaim that this information needs to be reviewed and 
approved by a clinician before it can be used. 

 
After you have generated the document, check back with the original input text to confirm that the generated summary 
accurately preserves the facts presented or discussed in the consultation. 
Repeat this validation twice to ensure the AI is not hallucinating. 

        """
    )
    client = AzureOpenAIWrapper()  # Use the wrapper class for clean interaction
    output = client.create_chat_completion(prompt)

    patient_summary_detailed = {"conversation": output}
    return patient_summary_detailed


if __name__ == '__main__':
    with open("input_file.json", "r") as f:
        json_text = json.load(f)
    summary = summarize_text(json_text)
    print(summary)
