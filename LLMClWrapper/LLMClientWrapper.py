#For timing operations and adding delays
import time
#For logging events, errors, informations
import logging
#Type hints for better code documentation
from typing import Optional, Dict, Any, List, Callable
#For timestamp operations
from datetime import datetime
#For asynchronous operations
import asyncio
#For creating decorators
from functools import wraps
#For adding jitter to retry delays
import random

#If don't call this, the log might not show up or show up in very default format
logging.basicConfig(level=logging.INFO #Python log has hierarchy: DEBUG<INFO<WARNING<ERROR<CRITICAL, set the threshold to "INFO" means to ignore debug but show everything else.
                    ,format='%(asctime)s - %(levelname)s -%(message)s') #Add timestamp(Ex: 2026-02-05, 14:00:00), severity level(INFO, ERROR,...), the actual text you wrote in your log.

#basicConfig sets the rules, logger as worker following the rules.
#Instead of create a generic log, this create a specific object (logger instance) for specific file
logger=logging.getLogger(__name__) #__name__ is a default python variable.
#1.if run file directly, __name__ is __main__
#2. If this file is imported, (database.py), then __name__ becomes database

class LLMClientWrapper:
    
    #Constructor
    def _init_(self,api_key:str,
               base_url:str="https://api.openai.com/v1",
               max_retries:int=3,
               initial_retry_delay:float=1.0,#how many seconds to wait before first rery
               max_retry_delay:float=60.0,#max number of requests allowed per minute
               rate_limit_tokens_per_minute:int= 10000,#Max tokens allowed per minute(API limit)
               requests_per_minute:int= 60):
        self.api_key=api_key
        self.base_url=base_url
        self.max_retries=max_retries
        self.initial_retry_delay=initial_retry_delay
        self.max_retry_delay=max_retry_delay
        self.rate_limit_tokens_per_minute=rate_limit_tokens_per_minute
        self.requests_per_minute=requests_per_minute
    
        #Token bucket algorithm implementation:
    """
    Imagine a physical bucket that holds "tokens."

    1.Token Generation: Tokens are added to the bucket at a constant, fixed rate (e.g., 5 tokens per second).

    2.Capacity: The bucket has a maximum size. If the bucket is full, new tokens are discarded.

    3.Processing: Every time a request (or packet) comes in, it must "grab" a token from the bucket to be processed.

    4.The "Burst": If the bucket is full because no requests have happened for a while, a large "burst" of requests can all be processed instantly until the bucket is empty. Once empty, the system must wait for new tokens to generate.

    Once the bucket is empty and a request is made, then the request is blocked and discarded.
    """
        self.token_bucket=rate_limit_tokens_per_minute
        #track timestamps of recent requests for request-based rate limiting
        self.requests_tokens=[]#Lists of timestamps when requests were made
        #Track when we last refilled the token bucket
        self.last_refill_time=time.time() #Current time in seconds since epoch
        self.stats=(
            "total_requests":0,#count of all requests made(include retries)
            "successful_requests":0,#count of requests that succeeded
            "failed_requests":0,#count of requests that failed after all retries
            "total_retries":0,#count of all retry attempts
            "rate_limit_hits":0 #count of times we hit rate limit
        )
        #log initialization message for debugging monitoring
        logger.info(f"LLMClientWrapper initialized with max_retries={max_retries},"
                    f"rate_limit={rate_limit_tokens_per_minute} tokens/min")
        
    #Refill the token bucket based on elapsed time since last refill
    #This is not a pure token bucket algorithm
    """
    For refill token bucket in token bucket:
    def refill_token_bucket(self):
        current_time = time.time()
        time_elapsed = current_time - self.last_refill_time
    
        # Calculate refill
        tokens_to_add = (time_elapsed / 60) * self.rate_limit_tokens_per_minute
    
        # Use a dedicated capacity variable if you want to allow bursts
        self.token_bucket = min(self.max_capacity, self.token_bucket + tokens_to_add)
    
        self.last_refill_time = current_time
        # NO NEED for self.requests_tokens list here!
    """
    def refill_token_bucket(self):
        current_time=time.time()
        #Calculate how much time has passed since last refill
        time_elapsed=current_time - self.last_refill_time
        #Calculate how many tokens to add based on time elapsed
        tokens_to_add=(time_elapsed/60)*self.rate_limit_tokens_per_minute
        #Add tokens to bucket, but don't exceed maximum capacity
        self.token_bucket=min(self.rate_limit_tokens_per_minute,
                              self.token_bucket+tokens_to_add)
        #Update last refill time to now
        self.last_refill_time=current_time
        #clean up old request timestamps - only keep those from teh last minute
        one_minute_ago=current_time-60
        #Use list comprehension to filter out timestamps older than 1 minute
        #This keeps our request tokens list from growing indefinitely
        self.requests_tokens=[t for t in self.requests_tokens if t>one_minute_ago]
        #Debug log to show token bucket status (only visible if logging level is DEBUG)
        logger.debug(f"Token bucket refilled: {self.token_bucket:.2f} tokens available")
        
    #Wait if necessary to respect rate limits before making a request
    def wait_for_rate_limit(self, required_tokens: int=1):     
        #required_tokens: number of tokens needed for the upcoming request
        #This is an estimate used for token based rate limiting
        
        #First refill the token bucket to account for time passed
        self.refill_token_bucket()
        
        #-- Token-based rate limit check --
        #Loop until we have enough tokens in the bucket
        while self.token_bucket < required_tokens:
            #Log why we are waiting(debug level so it doesn't spam production log)
            logger.debug(f"Insufficient tokens. Available: {self.token_bucket:.2f},"
                         f"Required: {required_tokens}")
            #Calculate how long to wait based on token deficit
            time_to_wait=(required_tokens-self.token_bucket)*60/self.rate_limit_tokens_per_minute
            #No less than 0.1 seconds, prevents busy-waiting
            time.sleep(max(0.1,time_to_wait))
            #After waiting, refill the bucket again
            self.refill_token_bucket()
            
    #-- Consume tokens and record request --
    #Check if we've made too many requests in the last minute
        if len(self.requests_tokens) >= self.requests_per_minute:
            #get the timestamp of teh oldest request in the last minute
            oldest_request= self.requests_tokens[0]
            #Calculate how long until that requeset is one minute old
            time_to_wait=60 - (time.time()-oldest_request) 
            #If we need to wait, wait that amount
            if time_to_wait>0:
                logger.debug(f"Request rate limit reached. Waiting {time_to_wait:.2f} seconds")
                time.sleep(time_to_wait)
                
        #-- Consume tokens and Record Request --
        #Deduct the estimated tokens from out bucket
        self.token_bucket-=required_tokens
        #Record the timestamp of this request for future rate limiting
        self.requests_tokens.append(time.time())
        #Log that we passed rate limiting checks
        logger.debug(f"Rate limit check passed. Tokens remaining: {self.token_bucket:.2f}")
        
    def _calculate_retry_delay(self, retry_count: int) -> float:
        """
        Calculate delay for exponential backoff with jitter.
        Exponential backoff means delays double each time.
        Jitter adds randomness to prevent synchronized retries (thundering herd problem).
        
        Args:
            retry_count: Current retry attempt number (0 = first retry, 1 = second retry, etc.)
            
        Returns:
            Delay in seconds before next retry
        """
        # Exponential backoff formula: delay = base_delay * 2^retry_count
        # Example: base=1, retry_count=2 → delay = 1 * 2^2 = 4 seconds
        delay = self.initial_retry_delay * (2 ** retry_count)
        
        # Add jitter: random factor between 0.8 and 1.2 (±20%)
        # This prevents many clients from retrying at exactly the same time
        jitter = random.uniform(0.8, 1.2)
        delay *= jitter  # Apply jitter to the delay
        
        # Cap the delay at the maximum allowed delay
        # Prevents waiting for extremely long times
        delay = min(delay, self.max_retry_delay)
        
        # Log the calculated delay for debugging
        logger.debug(f"Retry {retry_count + 1}: waiting {delay:.2f} seconds")
        
        return delay  # Return the calculated delay

    def _should_retry(self, status_code: int, error_type: Optional[str] = None) -> bool:
        """
        Determine if a request should be retried based on the error received.
        Not all errors should be retried (e.g., invalid API key, bad request).
        
        Args:
            status_code: HTTP status code from the API response
            error_type: Type of error received from the API (if provided)
            
        Returns:
            Boolean: True if we should retry, False if we should fail immediately
        """
        # Retry on these HTTP status codes:
        # 408 = Timeout
        # 429 = Rate Limit Exceeded (Too Many Requests)
        # 500 = Internal Server Error
        # 502 = Bad Gateway
        # 503 = Service Unavailable
        # 504 = Gateway Timeout
        if status_code in [408, 429, 500, 502, 503, 504]:
            return True  # These are retryable errors
        
        # List of error types that suggest temporary issues
        retryable_errors = [
            "timeout",  # Request timed out
            "rate_limit_exceeded",  # Hit rate limit
            "server_error",  # Server-side error
            "overloaded",  # Server is overloaded
            "temporarily_unavailable"  # Service temporarily unavailable
        ]
        
        # Check if the error_type contains any of the retryable error keywords
        # error_type might be None, so check if it exists first
        if error_type and any(err in error_type.lower() for err in retryable_errors):
            return True  # This is a retryable error type
            
        return False  # Not a retryable error

    def make_request_with_retry(
        self,
        endpoint: str,  # API endpoint like 'chat/completions' or 'completions'
        payload: Dict[str, Any],  # The data to send to the API
        estimated_tokens: int = 100,  # Estimate of tokens this request will use
        custom_headers: Optional[Dict[str, str]] = None  # Optional additional headers
    ) -> Dict[str, Any]:
        """
        Make an API request with automatic retries and rate limiting.
        This is the main method that applications will call.
        
        Args:
            endpoint: API endpoint (e.g., 'chat/completions')
            payload: Request payload containing model, messages, parameters, etc.
            estimated_tokens: Estimated tokens for rate limiting
            custom_headers: Additional headers to include in the request
            
        Returns:
            API response as dictionary
            
        Raises:
            Exception: If all retries fail
        """
        # Import requests here to avoid requiring it if not using sync methods
        import requests
        from requests.exceptions import RequestException  # For catching network errors
        
        # Construct the full URL for the API request
        url = f"{self.base_url}/{endpoint}"
        
        # Set up default headers required by most LLM APIs
        headers = {
            "Authorization": f"Bearer {self.api_key}",  # Authentication header
            "Content-Type": "application/json"  # Tell server we're sending JSON
        }
        
        # If custom headers were provided, add them to the headers dictionary
        if custom_headers:
            headers.update(custom_headers)  # Merge custom headers with default headers
        
        # Increment total requests counter for statistics
        self.stats["total_requests"] += 1
        
        # --- SAFE LOGGING OF REQUEST ---
        # Create a copy of the payload to avoid modifying the original
        safe_payload = payload.copy()
        
        # Check if the payload contains messages (common for chat APIs)
        if "messages" in safe_payload:
            # Redact message content in logs for privacy/security
            # Only include first 2 messages to keep logs readable
            safe_payload["messages"] = [
                {k: ("[REDACTED]" if k == "content" else v)  # Redact 'content' field
                 for k, v in msg.items()}  # For each key-value pair in the message
                for msg in safe_payload["messages"][:2]  # Only process first 2 messages
            ]
        
        # Log that we're making a request (INFO level - always visible)
        logger.info(f"Making request to {endpoint}")
        
        # Log the payload details (DEBUG level - only visible when debugging)
        logger.debug(f"Request payload: {safe_payload}")
        
        # --- RETRY LOOP ---
        # Try up to max_retries + 1 times (+1 for the initial attempt)
        for retry_count in range(self.max_retries + 1):
            try:
                # Apply rate limiting before each attempt
                # This will wait if we're hitting rate limits
                self._wait_for_rate_limit(estimated_tokens)
                
                # --- MAKE THE API CALL ---
                # Record start time to measure how long the request takes
                start_time = time.time()
                
                # Make the actual HTTP POST request to the API
                response = requests.post(
                    url=url,  # The full URL to send request to
                    headers=headers,  # Headers including auth
                    json=payload,  # The payload data (automatically converted to JSON)
                    timeout=30  # 30 second timeout - prevents hanging forever
                )
                
                # Calculate how long the request took
                request_time = time.time() - start_time
                
                # Log completion with status code and time taken
                logger.info(f"Request completed in {request_time:.2f}s - "
                          f"Status: {response.status_code}")
                
                # --- PROCESS SUCCESSFUL RESPONSE ---
                # Check if the request was successful (HTTP 200 OK)
                if response.status_code == 200:
                    # Parse the JSON response into a Python dictionary
                    response_data = response.json()
                    
                    # If the response includes token usage information, log it
                    if "usage" in response_data:
                        usage = response_data["usage"]  # Get usage dictionary
                        logger.info(f"Token usage - Prompt: {usage.get('prompt_tokens', 0)}, "
                                  f"Completion: {usage.get('completion_tokens', 0)}, "
                                  f"Total: {usage.get('total_tokens', 0)}")
                    
                    # Increment successful requests counter
                    self.stats["successful_requests"] += 1
                    
                    # Return the response data to the caller
                    return response_data
                
                # --- PROCESS ERROR RESPONSE ---
                else:
                    # Create a default error message
                    error_msg = f"API Error: {response.status_code}"
                    error_data = {}  # Will hold parsed error data if available
                    
                    try:
                        # Try to parse the error response as JSON
                        error_data = response.json()
                        
                        # Extract error message from the response structure
                        # LLM APIs usually have error.message field
                        error_msg = error_data.get("error", {}).get("message", error_msg)
                        
                        # Extract error type if available (e.g., "rate_limit_exceeded")
                        error_type = error_data.get("error", {}).get("type", "")
                    except:
                        # If response isn't valid JSON, use default error type
                        error_type = "unknown"
                    
                    # Log the error (WARNING level - indicates something went wrong)
                    logger.warning(f"API error: {error_msg} (Status: {response.status_code})")
                    
                    # --- DECIDE WHETHER TO RETRY ---
                    # Check if we have retries left AND if this error is retryable
                    if retry_count < self.max_retries and self._should_retry(
                        response.status_code, error_type
                    ):
                        # Check if this was a rate limit error
                        if "rate_limit" in error_msg.lower() or response.status_code == 429:
                            # Increment rate limit hit counter
                            self.stats["rate_limit_hits"] += 1
                            logger.warning("Rate limit hit. Will retry.")
                        
                        # Increment retry counter
                        self.stats["total_retries"] += 1
                        
                        # Calculate how long to wait before retrying
                        delay = self._calculate_retry_delay(retry_count)
                        
                        # Wait before trying again
                        time.sleep(delay)
                        
                        # Continue to next iteration of retry loop
                        continue
                    else:
                        # If we shouldn't retry or have no retries left, this is a final failure
                        self.stats["failed_requests"] += 1
                        
                        # Raise an exception with descriptive message
                        raise Exception(f"API request failed after {retry_count + 1} attempts: {error_msg}")
                        
            # --- HANDLE NETWORK ERRORS ---
            except RequestException as e:
                # This catches network-level errors (no connection, DNS failure, etc.)
                logger.warning(f"Network error on attempt {retry_count + 1}: {str(e)}")
                
                # Check if we should retry
                if retry_count < self.max_retries:
                    # Increment retry counter
                    self.stats["total_retries"] += 1
                    
                    # Calculate and wait for retry delay
                    delay = self._calculate_retry_delay(retry_count)
                    time.sleep(delay)
                    
                    # Continue to next retry attempt
                    continue
                else:
                    # No retries left, record failure and raise exception
                    self.stats["failed_requests"] += 1
                    raise Exception(f"Network error after {retry_count + 1} attempts: {str(e)}")
            
            # --- HANDLE UNEXPECTED ERRORS ---
            except Exception as e:
                # Catch-all for any other unexpected errors
                logger.error(f"Unexpected error: {str(e)}")
                
                # Record as failed request
                self.stats["failed_requests"] += 1
                
                # Re-raise the exception
                raise
        
        # This line should never be reached because we raise exceptions above
        # But it's here as a safety net
        raise Exception("Max retries exceeded")

    # The rest of the class continues with async methods and convenience methods...
    # Each follows similar patterns with detailed error handling and logging

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics about API usage.
        
        Returns:
            Dictionary with statistics counters
        """
        # Return a copy to prevent external code from modifying our internal stats
        return self.stats.copy()
