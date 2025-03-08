import os
import time
import json
import argparse
from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class AIAppGenerator:
    def __init__(self, api_key: str, site_url: str = "https://example.com", site_name: str = "AI App Generator", 
                 teacher_model: str = "anthropic/claude-3-opus", generator_model: str = "openai/gpt-4o", 
                 verbose: bool = False):
        """
        Initialize the AI Application Generator with the specified models.
        
        Args:
            api_key: OpenRouter API key
            site_url: Your site URL for OpenRouter rankings
            site_name: Your site name for OpenRouter rankings
            teacher_model: Model to use for the teacher role
            generator_model: Model to use for the generator role
            verbose: Whether to print detailed progress information
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        self.extra_headers = {
            "HTTP-Referer": site_url,
            "X-Title": site_name,
        }
        
        self.teacher_model = teacher_model
        self.generator_model = generator_model
        self.verbose = verbose
        
        # Conversation history for each role
        self.teacher_conversation = []
        self.generator_conversation = []
        
        # Project details
        self.project_name = ""
        self.project_description = ""
        self.detailed_requirements = ""
        self.current_code = ""
        
        self.log(f"Initialized with teacher: {teacher_model}, generator: {generator_model}")
    
    def log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def call_model(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, 
                  max_tokens: int = 4000) -> str:
        """
        Call the specified model with the given messages.
        
        Args:
            model: Model identifier string
            messages: List of message dictionaries (role and content)
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The model's response text
        """
        self.log(f"Calling {model}...")
        
        # Adding retry mechanism
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    extra_headers=self.extra_headers,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                # Safely extract the content
                if completion and hasattr(completion, 'choices') and len(completion.choices) > 0:
                    message = completion.choices[0].message
                    if message and hasattr(message, 'content'):
                        response = message.content
                        if response is not None:
                            self.log(f"Received {len(response)} chars response")
                            return response
                        else:
                            self.log("Warning: Received None content from API")
                            return "Error: Empty response from the model. Please try again."
                    else:
                        self.log("Warning: Response message has no content attribute")
                else:
                    self.log("Warning: Invalid response structure from API")
                
                # If we reach here, something went wrong with the response format
                error_msg = "Error: Unexpected API response format. Please check your API key and try again."
                self.log(error_msg)
                
                if attempt < max_retries - 1:
                    self.log(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    return error_msg
                    
            except Exception as e:
                self.log(f"Error calling API: {str(e)}")
                
                if attempt < max_retries - 1:
                    self.log(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    return f"Error: API request failed after {max_retries} attempts. Last error: {str(e)}"
        
        return "Error: Failed to get a valid response from the model."
    
    def initialize_project(self, project_description: str) -> str:
        """
        Initialize the project with a basic description.
        
        Args:
            project_description: Initial user description of the project
            
        Returns:
            The teacher's initial response with questions and suggestions
        """
        self.project_description = project_description
        
        # Validate input
        if not project_description or len(project_description.strip()) < 5:
            return "Error: Please provide a more detailed project description (at least a few words)."
        
        # Initialize teacher conversation
        self.teacher_conversation = [
            {"role": "system", "content": "You are a senior software architect and requirements analyst. Your job is to help refine project requirements by asking clarifying questions and providing structured, detailed specifications. Be concise but thorough, focusing on technical requirements, architecture, features, and potential challenges."},
            {"role": "user", "content": f"I need help developing this application: {project_description}\n\nPlease help me refine these requirements by asking clarifying questions and suggesting improvements or considerations I might have missed. Structure your response clearly with sections for different aspects of the application."}
        ]
        
        # Get initial teacher response
        teacher_response = self.call_model(self.teacher_model, self.teacher_conversation)
        
        # Only append to conversation history if we got a valid response
        if not teacher_response.startswith("Error:"):
            self.teacher_conversation.append({"role": "assistant", "content": teacher_response})
        
        return teacher_response
    
    def refine_requirements(self, user_response: str) -> str:
        """
        Continue the requirements refinement dialogue with the teacher.
        
        Args:
            user_response: User's response to the teacher's previous message
            
        Returns:
            The teacher's next response
        """
        # Validate input
        if not user_response or len(user_response.strip()) < 2:
            return "Error: Please provide a more detailed response to continue the conversation."
        
        # Add user response to teacher conversation
        self.teacher_conversation.append({"role": "user", "content": user_response})
        
        # Get next teacher response
        teacher_response = self.call_model(self.teacher_model, self.teacher_conversation)
        
        # Only append to conversation history if we got a valid response
        if not teacher_response.startswith("Error:"):
            self.teacher_conversation.append({"role": "assistant", "content": teacher_response})
        
        return teacher_response
    
    def finalize_requirements(self, user_response: str) -> str:
        """
        Finalize the project requirements based on the dialogue.
        
        Args:
            user_response: User's final response confirming requirements
            
        Returns:
            Structured detailed requirements document
        """
        prompt = f"""Based on our discussion, please create a final, structured requirements document for the application. 
        Include:
        1. Project overview
        2. Core features and functionality
        3. Technical specifications (languages, frameworks, libraries)
        4. Architecture overview
        5. Data models (if applicable)
        6. API endpoints (if applicable)
        7. UI/UX requirements (if applicable)

        User's final input: {user_response}
        """
        
        self.teacher_conversation.append({"role": "user", "content": prompt})
        detailed_requirements = self.call_model(self.teacher_model, self.teacher_conversation)
        
        # Only append to conversation history if we got a valid response
        if not detailed_requirements.startswith("Error:"):
            self.teacher_conversation.append({"role": "assistant", "content": detailed_requirements})
            self.detailed_requirements = detailed_requirements
        
        return detailed_requirements
    
    def generate_initial_code(self) -> str:
        """
        Generate the initial code based on the detailed requirements.
        
        Returns:
            The initial code for the application
        """
        # Check if we have detailed requirements
        if not self.detailed_requirements:
            return "Error: No detailed requirements available. Please finalize requirements first."
        
        # Initialize generator conversation
        self.generator_conversation = [
            {"role": "system", "content": "You are an expert software developer capable of converting detailed requirements into working code. Focus on producing clean, well-structured, functional code with proper organization, error handling, and comments. Generate complete, executable code that follows best practices for the specified languages and frameworks."},
            {"role": "user", "content": f"Please generate complete, executable code for the following application requirements:\n\n{self.detailed_requirements}\n\nProvide all necessary files, directory structure, and setup instructions. Ensure the code is well-documented with comments explaining key components and logic."}
        ]
        
        # Get initial code generation
        generated_code = self.call_model(self.generator_model, self.generator_conversation, temperature=0.1, max_tokens=8000)
        
        # Only append to conversation history if we got a valid response
        if not generated_code.startswith("Error:"):
            self.generator_conversation.append({"role": "assistant", "content": generated_code})
            self.current_code = generated_code
        
        return generated_code
    
    def request_code_improvements(self, feedback: str = "") -> str:
        """
        Request improvements to the current code from the teacher.
        
        Args:
            feedback: Optional user feedback to guide improvements
            
        Returns:
            Improvement suggestions from the teacher
        """
        # Check if we have code to improve
        if not self.current_code:
            return "Error: No code available to improve. Please generate initial code first."
        
        prompt = f"""
        Please analyze the following code and suggest specific improvements:
        
        {self.current_code}
        
        User feedback: {feedback}
        
        Focus on:
        1. Code quality and structure
        2. Performance optimizations
        3. Better libraries or approaches
        4. Security concerns
        5. Scalability issues
        6. Error handling
        7. Documentation improvements
        8. Testing approaches
        
        Provide specific, actionable suggestions with code examples where appropriate.
        """
        
        self.teacher_conversation.append({"role": "user", "content": prompt})
        improvement_suggestions = self.call_model(self.teacher_model, self.teacher_conversation)
        
        # Only append to conversation history if we got a valid response
        if not improvement_suggestions.startswith("Error:"):
            self.teacher_conversation.append({"role": "assistant", "content": improvement_suggestions})
        
        return improvement_suggestions
    
    def implement_improvements(self, improvements: str) -> str:
        """
        Implement the suggested improvements to the code.
        
        Args:
            improvements: Improvement suggestions from the teacher
            
        Returns:
            The improved code
        """
        # Check if we have code to improve
        if not self.current_code:
            return "Error: No code available to improve. Please generate initial code first."
        
        # Check if we have valid improvements
        if improvements.startswith("Error:"):
            return "Error: Cannot implement invalid improvement suggestions."
        
        prompt = f"""
        Please implement the following suggested improvements to the current code:
        
        CURRENT CODE:
        {self.current_code}
        
        SUGGESTED IMPROVEMENTS:
        {improvements}
        
        Provide the complete, improved code with all changes implemented. Maintain the overall structure and functionality while incorporating the improvements.
        """
        
        self.generator_conversation.append({"role": "user", "content": prompt})
        improved_code = self.call_model(self.generator_model, self.generator_conversation, temperature=0.1, max_tokens=8000)
        
        # Only append to conversation history if we got a valid response
        if not improved_code.startswith("Error:"):
            self.generator_conversation.append({"role": "assistant", "content": improved_code})
            self.current_code = improved_code
        
        return improved_code
    
    def save_project(self, output_dir: str = "output") -> str:
        """
        Save the project files to disk.
        
        Args:
            output_dir: Directory to save the project files
            
        Returns:
            A message indicating success or failure
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the requirements document if available
            if self.detailed_requirements and not self.detailed_requirements.startswith("Error:"):
                with open(os.path.join(output_dir, "requirements.md"), "w", encoding="utf-8") as f:
                    f.write(self.detailed_requirements)
            
            # Save the conversation history
            with open(os.path.join(output_dir, "teacher_conversation.json"), "w", encoding="utf-8") as f:
                json.dump(self.teacher_conversation, f, indent=2, ensure_ascii=False)
            
            with open(os.path.join(output_dir, "generator_conversation.json"), "w", encoding="utf-8") as f:
                json.dump(self.generator_conversation, f, indent=2, ensure_ascii=False)
            
            # Extract and save code files if available
            if self.current_code and not self.current_code.startswith("Error:"):
                self._extract_and_save_code_files(self.current_code, output_dir)
            else:
                # Save any error messages
                with open(os.path.join(output_dir, "errors.txt"), "w", encoding="utf-8") as f:
                    f.write(self.current_code if self.current_code else "No code was generated")
            
            self.log(f"Project saved to {output_dir}")
            return f"Project successfully saved to '{output_dir}' directory."
            
        except Exception as e:
            error_msg = f"Error saving project: {str(e)}"
            self.log(error_msg)
            return error_msg
    
    def _extract_and_save_code_files(self, code_text: str, output_dir: str) -> None:
        """
        Extract and save individual code files from the generated code text.
        
        Args:
            code_text: The raw code text containing file blocks
            output_dir: Directory to save the extracted files
        """
        # First, try to find files marked with markdown-style code blocks
        import re
        
        # Pattern for markdown code blocks with filenames: ```language:filename or ```filename
        file_blocks = re.findall(r'```(?:(\w+)\s*:?\s*)?([^\n:]+)?\n(.*?)```', code_text, re.DOTALL)
        files_saved = 0
        
        if file_blocks:
            for match in file_blocks:
                language, filename, content = match
                
                # If filename is missing but language is present, it might be a language specifier
                if not filename and language and language not in ['python', 'bash', 'javascript', 'typescript', 'java', 'html', 'css']:
                    filename = language
                
                # Skip code blocks without filenames or with common language specifiers only
                if filename and not filename.strip().startswith('output') and filename.strip() not in ['python', 'bash', 'javascript', 'typescript', 'java', 'html', 'css']:
                    try:
                        # Create subdirectories if needed
                        filepath = os.path.join(output_dir, filename.strip())
                        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                        
                        with open(filepath, 'w', encoding="utf-8") as f:
                            f.write(content)
                        self.log(f"Saved file: {filepath}")
                        files_saved += 1
                    except Exception as e:
                        self.log(f"Error saving file {filename}: {str(e)}")
        
        # If no structured file blocks found or no files were successfully saved, save everything as a single file
        if files_saved == 0:
            with open(os.path.join(output_dir, "generated_code.py"), "w", encoding="utf-8") as f:
                f.write(code_text)
            self.log(f"Saved all code to generated_code.py")


def interactive_mode(generator: AIAppGenerator):
    """Run the application generator in interactive mode."""
    print("\n===== AI Application Generator =====\n")
    
    # Step 1: Get initial project description
    print("Please describe the application you want to build:\n")
    project_description = input("> ")
    
    # Step 2: Initialize project and start requirements refinement
    print("\nInitializing project and analyzing requirements...\n")
    teacher_response = generator.initialize_project(project_description)
    
    if teacher_response.startswith("Error:"):
        print(f"\n{teacher_response}")
        print("\nPlease restart the program with a more detailed project description.")
        return
        
    print("\nTeacher's Initial Analysis:\n")
    print(teacher_response)
    
    # Step 3: Refine requirements through dialogue
    while True:
        print("\nRespond to continue refining requirements, or type 'DONE' to proceed to code generation:\n")
        user_response = input("> ")
        
        if user_response.strip().upper() == "DONE":
            break
        
        print("\nProcessing your response...\n")
        teacher_response = generator.refine_requirements(user_response)
        
        if teacher_response.startswith("Error:"):
            print(f"\n{teacher_response}")
            continue
            
        print("\nTeacher's Response:\n")
        print(teacher_response)
    
    # Step 4: Finalize requirements
    print("\nFinalizing requirements. Please provide any final details or confirmations:\n")
    final_input = input("> ")
    
    print("\nGenerating detailed requirements document...\n")
    detailed_requirements = generator.finalize_requirements(final_input)
    
    if detailed_requirements.startswith("Error:"):
        print(f"\n{detailed_requirements}")
        print("\nPlease try again with more specific requirements.")
        return
        
    print("\nDetailed Requirements Document:\n")
    print(detailed_requirements)
    
    # Step 5: Generate initial code
    print("\nGenerating initial code based on requirements...\n")
    initial_code = generator.generate_initial_code()
    
    if initial_code.startswith("Error:"):
        print(f"\n{initial_code}")
        print("\nPlease try refining the requirements and try again.")
        return
        
    print("Initial code generated. Here's a preview (first 500 chars):\n")
    preview = initial_code[:500] + "..." if len(initial_code) > 500 else initial_code
    print(preview + "\n")
    
    # Step 6: Iterative improvement
    improvement_iteration = 1
    while True:
        print(f"\n=== Improvement Iteration {improvement_iteration} ===\n")
        print("Provide feedback for improvements, or type 'DONE' to finalize:\n")
        feedback = input("> ")
        
        if feedback.strip().upper() == "DONE":
            break
        
        print("\nRequesting improvement suggestions...\n")
        suggestions = generator.request_code_improvements(feedback)
        
        if suggestions.startswith("Error:"):
            print(f"\n{suggestions}")
            continue
            
        print("Improvement suggestions:\n")
        print(suggestions)
        
        print("\nImplementing improvements...\n")
        improved_code = generator.implement_improvements(suggestions)
        
        if improved_code.startswith("Error:"):
            print(f"\n{improved_code}")
            continue
            
        print("Improvements implemented. Here's a preview (first 500 chars):\n")
        preview = improved_code[:500] + "..." if len(improved_code) > 500 else improved_code
        print(preview + "\n")
        
        improvement_iteration += 1
    
    # Step 7: Save the project
    print("\nSaving project files...\n")
    result = generator.save_project()
    print(result + "\n")
    print("Thank you for using the AI Application Generator!")


def test_api_connection(api_key, model="anthropic/claude-3-haiku", verbose=True):
    """Test the API connection with a simple request."""
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        if verbose:
            print(f"Testing API connection with model: {model}...")
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://example.com",
                "X-Title": "API Connection Test",
            },
            model=model,
            messages=[{"role": "user", "content": "Hi, this is a test. Respond with 'Connection successful!' if you receive this."}],
            max_tokens=20,
        )
        
        if completion and hasattr(completion, 'choices') and len(completion.choices) > 0:
            message = completion.choices[0].message
            if message and hasattr(message, 'content') and message.content:
                if verbose:
                    print(f"Connection successful! Response: {message.content}")
                return True
        
        if verbose:
            print("API connection test failed: Unexpected response format")
        return False
        
    except Exception as e:
        if verbose:
            print(f"API connection test failed: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="AI Application Generator")
    parser.add_argument("--api-key", type=str, help="OpenRouter API key")
    parser.add_argument("--teacher-model", type=str, default="google/gemini-2.0-pro-exp-02-05:free", 
                      help="Model to use for the teacher role")
    parser.add_argument("--generator-model", type=str, default="deepseek/deepseek-r1-zero:free", 
                      help="Model to use for the generator role")
    parser.add_argument("--site-url", type=str, default="https://example.com", 
                      help="Your site URL for OpenRouter rankings")
    parser.add_argument("--site-name", type=str, default="AI App Generator", 
                      help="Your site name for OpenRouter rankings")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--test-api", action="store_true", help="Test API connection and exit")
    
    args = parser.parse_args()
    
    # Use API key from arguments or environment
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OpenRouter API key is required. Provide it via --api-key or set the OPENROUTER_API_KEY environment variable.")
        return 1
    
    # Test API connection if requested
    if args.test_api:
        success = test_api_connection(api_key, model="anthropic/claude-3-haiku", verbose=True)
        return 0 if success else 1
    
    # Validate the API connection before proceeding
    print("Testing API connection before starting...")
    if not test_api_connection(api_key, model="anthropic/claude-3-haiku", verbose=args.verbose):
        print("\nError: Could not connect to the OpenRouter API. Please check your API key and internet connection.")
        return 1
    
    # Create the generator
    generator = AIAppGenerator(
        api_key=api_key,
        teacher_model=args.teacher_model,
        generator_model=args.generator_model,
        site_url=args.site_url,
        site_name=args.site_name,
        verbose=args.verbose
    )
    
    try:
        # Run interactive mode
        interactive_mode(generator)
        return 0
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
        return 0
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)