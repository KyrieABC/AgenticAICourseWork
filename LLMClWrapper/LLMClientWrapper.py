import time
import logging
from openai import OpenAI,RateLimitError, APIStatusError, APIConnectionError
import os
from dotenv import load_dotenv
import random # Useful for adding "jitter" to retries

#If don't call this, the log might not show up or show up in very default format
logging.basicConfig(
    level=logging.INFO,#Python log has hierarchy: DEBUG<INFO<WARNING<ERROR<CRITICAL, set the threshold to "INFO" means to ignore debug but show everything else.
    format='%(asctime)s - %(levelname)s - %(message)s'
)#Add timestamp(Ex: 2026-02-05, 14:00:00), severity level(INFO, ERROR,...), the actual text you wrote in your log.

#basicConfig sets the rules, logger as worker following the rules.
#Instead of create a generic log, this create a specific object (logger instance) for specific file
logger = logging.getLogger(__name__)
#1.if run file directly, __name__ is __main__
#2. If this file is imported, (database.py), then __name__ becomes database

class LLMClientWrapper:
    def __init__(self, api_key, model="gpt-3.5-turbo", requests_per_minute=3):
        # Using self. (varName) = parameter as we discussed!
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        # Rate Limiting (Token Bucket)
        self.max_capacity = requests_per_minute
        self.tokens = requests_per_minute
        self.rate_per_sec = requests_per_minute / 60.0
        self.last_refill_time = time.time()

    def _refill(self):
        """Internal method to update token count based on time passed."""
        now = time.time()
        passed = now - self.last_refill_time
        
        # Add tokens: (time * rate)
        new_tokens = passed * self.rate_per_sec
        self.tokens = min(self.max_capacity, self.tokens + new_tokens)
        self.last_refill_time = now

    def generate_with_retry(self, prompt, max_retries=3):
        """Generates a response with exponential backoff for transient errors."""
        retries = 0
        wait_time = 1  # Start with a 1-second wait

        while retries <= max_retries:
            # 1. Check our local Token Bucket first
            self._refill()
            if self.tokens < 1:
                logger.warning(f"Local rate limit hit. Waiting {wait_time}s to retry...")
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2 # Exponential increase
                continue

            try:
                self.tokens -= 1
                logger.info(f"Attempt {retries + 1}: Sending request...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content

            except RateLimitError as e:
                # 2. Handle "429 Too Many Requests" from the Provider
                logger.warning(f"Provider Rate Limit: {e}. Retrying...")
                
            except (APIConnectionError, APIStatusError) as e:
                # 3. Handle "503 Service Unavailable" or Network Blips
                if hasattr(e, 'status_code') and e.status_code >= 500:
                    logger.warning(f"Server Error ({e.status_code}): {e}. Retrying...")
                else:
                    # 4. Handle Fatal Errors (401 Invalid Key, 400 Bad Request)
                    logger.error(f"FATAL ERROR: {e}")
                    raise  # Don't retry if the API key is wrong!

            # If we reach here, an error occurred that is worth retrying
            retries += 1
            if retries <= max_retries:
                # Add a little "jitter" (randomness) so multiple workers don't 
                # all retry at the exact same millisecond.
                sleep_with_jitter = wait_time + random.uniform(0, 1)
                logger.info(f"Retrying in {sleep_with_jitter:.2f} seconds...")
                time.sleep(sleep_with_jitter)
                wait_time *= 2  # Double the wait for next time

        logger.error("Max retries exceeded. Request failed.")
        return None


    def generate(self, prompt):
        """The main method to get a response from the AI."""
        self._refill()

        # Check: If tokens >= 1, subtract and allow. If not, block.
        if self.tokens < 1:
            logger.warning("Rate limit reached! Request discarded.")
            return None

        self.tokens -= 1
        
        try:
            logger.info(f"Sending request to {self.model}...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            # Simplify the output (Transformation)
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"API Call Failed: {e}")
            return None

load_dotenv(override=True)
wrapper=LLMClientWrapper(api_key=os.getenv("OPENAI_API_KEY"),model="gpt-4o-mini",requests_per_minute=5)
response=wrapper.generate("What's the colors of the American Flag?")
print(response)
# --- Usage Example ---
# wrapper = LLMClientWrapper(api_key="your-key-here", requests_per_minute=5)
# response = wrapper.generate("Why is the sky blue?")
# print(response)