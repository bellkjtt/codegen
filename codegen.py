import logging
from typing import List, Dict, Tuple, Any
from openai import AsyncOpenAI
import asyncio
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import traceback
import sys
import os
import tempfile
import shutil

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class DebuggingAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def debug_code(self, code: str, error_message: str, execution_context: str) -> str:
        prompt = f"""
        Debug and fix the following Python code that produced an error during execution.
        
        Code with error:
        {code}
        
        Error message:
        {error_message}
        
        Execution context:
        {execution_context}
        
        Please:
        1. Analyze the error
        2. Identify the root cause
        3. Fix the code
        4. Add additional error handling if needed
        5. Add logging statements for debugging
        
        Return only the fixed code without any explanation.
        """
        client = AsyncOpenAI(api_key=self.api_key)
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Python debugging expert who fixes runtime errors."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    async def verify_fix(self, original_error: str, fixed_code: str) -> Tuple[bool, str]:
        prompt = f"""
        Verify if the fixed code addresses the original error:
        
        Original error:
        {original_error}
        
        Fixed code:
        {fixed_code}
        
        Return only "YES" if the fix addresses the error, or "NO" with a brief explanation if it doesn't.
        """
        client = AsyncOpenAI(api_key=self.api_key)
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a code review expert verifying bug fixes."},
                {"role": "user", "content": prompt}
            ]
        )
        result = response.choices[0].message.content.strip().upper()
        return result.startswith("YES"), result

class RuntimeValidator:
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.globals_dict = {}

    def prepare_environment(self):
        sys.path.insert(0, self.temp_dir)

    def cleanup_environment(self):
        sys.path.remove(self.temp_dir)
        for key in list(self.globals_dict.keys()):
            del self.globals_dict[key]

    async def validate_runtime(self, code: str, test_data: str) -> Tuple[bool, str, Any]:
        self.prepare_environment()
        try:
            output_buffer = StringIO()
            error_buffer = StringIO()
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                exec(test_data, self.globals_dict)
                exec(code, self.globals_dict)
            return True, "", output_buffer.getvalue()
        except Exception as e:
            return False, f"{str(e)}\n{traceback.format_exc()}", error_buffer.getvalue()
        finally:
            self.cleanup_environment()

class PipelineExecutor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.debugging_agent = DebuggingAgent(api_key)

    async def execute_pipeline(self, combined_code: str) -> Tuple[bool, str, List[Dict[str, Any]]]:
        temp_dir = tempfile.mkdtemp()
        validator = RuntimeValidator(temp_dir)
        execution_results = []
        try:
            steps = self._split_code_into_steps(combined_code)
            for i, step in enumerate(steps, 1):
                while True:
                    success, error_msg, output = await validator.validate_runtime(step['code'], step['test_data'])
                    if success:
                        break
                    step['code'] = await self._fix_step(step['code'], error_msg, output)
                execution_results.append({
                    'step': i,
                    'success': success,
                    'error': error_msg,
                    'output': output,
                    'code': step['code']
                })
            return True, "", execution_results
        except Exception as e:
            return False, str(e), execution_results
        finally:
            shutil.rmtree(temp_dir)

    async def _fix_step(self, code: str, error_msg: str, context: str) -> str:
        while True:
            try:
                fixed_code = await self.debugging_agent.debug_code(code, error_msg, context)
                is_valid, verify_msg = await self.debugging_agent.verify_fix(error_msg, fixed_code)
                fixed_code = fixed_code.replace("```python", "").replace("```", "").strip()
                print(fixed_code,"고쳐진 코드")
                if is_valid:
                    return fixed_code
                else:
                    code = fixed_code
            except Exception:
                pass

    def _split_code_into_steps(self, combined_code: str) -> List[Dict[str, str]]:
        steps = []
        current_step = {'code': '', 'test_data': ''}
        lines = combined_code.split('\n')
        for line in lines:
            if line.startswith('# Step'):
                if current_step['code']:
                    steps.append(current_step.copy())
                current_step = {'code': '', 'test_data': ''}
            elif line.startswith('# Test data'):
                current_step['test_data'] += line + '\n'
            else:
                current_step['code'] += line + '\n'
        if current_step['code']:
            steps.append(current_step)
        return steps

class MultiStepSystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.pipeline_executor = PipelineExecutor(api_key)

    async def _generate_code(self, step_descriptions: List[str]) -> str:
        prompt = f"""
        Generate Python code for a data processing pipeline with the following steps:
        
        {chr(10).join(f'{i+1}. {desc}' for i, desc in enumerate(step_descriptions))}
        
        For each step:
        1. Include clear comments
        2. Add appropriate error handling
        3. Include logging statements
        4. Add test data and assertions
        
        Format each step as:
        # Step X
        [code]
        # Test data
        [test data]
        
        Use pandas and scikit-learn libraries where appropriate.
        """
        client = AsyncOpenAI(api_key=self.api_key)
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Python expert who generates production-ready code."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    async def process_steps(self, step_descriptions: List[str]) -> str:
        combined_code = await self._generate_code(step_descriptions)
        success, error_msg, results = await self.pipeline_executor.execute_pipeline(combined_code)
        if not success:
            print("Pipeline validation failed.")
            return "Pipeline validation failed."
        print("Pipeline validation succeeded.")
        return self._update_combined_code(combined_code, results)

    def _update_combined_code(self, original_code: str, results: List[Dict[str, Any]]) -> str:
        updated_code = original_code
        for result in results:
            if result['success'] and result['code']:
                step_marker = f"# Step {result['step']}"
                start_idx = updated_code.find(step_marker)
                if start_idx != -1:
                    next_step_marker = f"# Step {result['step'] + 1}"
                    end_idx = updated_code.find(next_step_marker)
                    if end_idx == -1:
                        end_idx = len(updated_code)
                    updated_code = (
                        updated_code[:start_idx] +
                        f"{step_marker}\n{result['code']}\n" +
                        updated_code[end_idx:]
                    )
        return updated_code

async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    system = MultiStepSystem(api_key)
    step_descriptions = [
        "Load model from storage or cloud service",  # 모델을 저장소나 클라우드에서 불러오기
        "Initialize model with API key",             # API 키로 모델 초기화
        "Verify model loaded successfully",          # 모델 로딩 성공 확인
        "Create API endpoint to interact with model", # 모델과 상호작용할 API 엔드포인트 생성
        "Test the API with sample requests",         # 샘플 요청으로 API 테스트
        "Return final response from model",          # 모델에서 최종 응답 반환
    ]
    final_code = await system.process_steps(step_descriptions)
    with open("generated_pipeline.py", "w") as f:
        f.write(final_code)
    print("Success" if "Pipeline validation succeeded" in final_code else "Failure")

if __name__ == "__main__":
    asyncio.run(main())
