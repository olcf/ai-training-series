"""Unit tests for RHAPSODY Task API.

Tests cover:
- Automatic UID generation
- Task validation (ComputeTask, AITask)
- Task serialization (to_dict, from_dict)
- Custom fields support
"""

import threading

import pytest

import rhapsody
from rhapsody import AITask
from rhapsody import ComputeTask
from rhapsody import TaskValidationError


class TestUIDGeneration:
    """Tests for automatic UID generation."""

    def test_auto_uid_generation(self):
        """Test UIDs are auto-generated with correct format."""
        task1 = ComputeTask(executable="/bin/echo")
        task2 = ComputeTask(executable="/bin/hostname")

        assert task1.uid.startswith("task.")
        assert task2.uid.startswith("task.")
        assert task1.uid != task2.uid

    def test_uid_format(self):
        """Test UID format is task.NNNNNN."""
        task = ComputeTask(executable="/bin/echo")

        # Format: task.NNNNNN (e.g., task.000001)
        assert task.uid.startswith("task.")
        uid_number = task.uid.split(".")[-1]
        assert len(uid_number) == 6
        assert uid_number.isdigit()

    def test_uid_counter_increments(self):
        """Test UID counter increments correctly."""
        task1 = ComputeTask(executable="/bin/echo")
        task2 = ComputeTask(executable="/bin/echo")

        uid1_num = int(task1.uid.split(".")[-1])
        uid2_num = int(task2.uid.split(".")[-1])

        assert uid2_num == uid1_num + 1

    def test_uid_thread_safety(self):
        """Test UID generation is thread-safe."""
        uids = []
        lock = threading.Lock()

        def create_tasks():
            local_uids = []
            for _ in range(100):
                task = ComputeTask(executable="/bin/echo")
                local_uids.append(task.uid)

            with lock:
                uids.extend(local_uids)

        threads = [threading.Thread(target=create_tasks) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All UIDs should be unique (1000 tasks from 10 threads)
        assert len(uids) == 1000
        assert len(uids) == len(set(uids))

    def test_manual_uid_override(self):
        """Test manual UID specification still works."""
        task = ComputeTask(uid="custom_id", executable="/bin/echo")
        assert task.uid == "custom_id"

    def test_uid_across_task_types(self):
        """Test UID generation works across different task types."""
        compute = ComputeTask(executable="/bin/echo")
        ai = AITask(prompt="test", model="gpt-4")

        assert compute.uid.startswith("task.")
        assert ai.uid.startswith("task.")
        assert compute.uid != ai.uid

    def test_mixed_auto_and_manual_uids(self):
        """Test mixing auto-generated and manual UIDs."""
        task1 = ComputeTask(executable="/bin/echo")  # Auto
        task2 = ComputeTask(uid="manual_id", executable="/bin/echo")  # Manual
        task3 = ComputeTask(executable="/bin/echo")  # Auto

        assert task1.uid.startswith("task.")
        assert task2.uid == "manual_id"
        assert task3.uid.startswith("task.")
        assert task1.uid != task3.uid


class TestComputeTaskValidation:
    """Tests for ComputeTask validation."""

    def test_executable_only_valid(self):
        """Test executable-only task is valid."""
        task = ComputeTask(executable="/bin/echo")
        assert task.executable == "/bin/echo"
        assert task.function is None

    def test_executable_with_arguments(self):
        """Test executable with arguments."""
        task = ComputeTask(executable="/bin/echo", arguments=["hello", "world"])
        assert task.executable == "/bin/echo"
        assert task.arguments == ["hello", "world"]

    def test_function_only_valid(self):
        """Test function-only task is valid."""

        def my_func():
            return 42

        task = ComputeTask(function=my_func)
        assert task.function == my_func
        assert task.executable is None

    def test_function_with_args_kwargs(self):
        """Test function with args and kwargs."""

        def my_func(a, b, c=10):
            return a + b + c

        task = ComputeTask(function=my_func, args=(5, 10), kwargs={"c": 20})
        assert task.function == my_func
        assert task.args == (5, 10)
        assert task.kwargs == {"c": 20}

    def test_both_executable_and_function_invalid(self):
        """Test both executable and function raises error."""
        with pytest.raises(TaskValidationError) as exc:
            ComputeTask(executable="/bin/echo", function=lambda: None)

        assert "cannot have both" in str(exc.value).lower()

    def test_neither_executable_nor_function_invalid(self):
        """Test missing both executable and function raises error."""
        with pytest.raises(TaskValidationError) as exc:
            ComputeTask()

        assert "requires either" in str(exc.value).lower()


class TestBackendField:
    """Tests for the 'backend' field in tasks."""

    def test_compute_task_with_backend(self):
        """Test that ComputeTask supports explicit backend."""
        task = ComputeTask(executable="/bin/echo", backend="dragon")
        assert task.backend == "dragon"
        task = ComputeTask(executable="/bin/echo", backend="dragon")
        assert task.backend == "dragon"
        assert task["backend"] == "dragon"

    def test_ai_task_with_backend(self):
        """Test that AITask supports explicit backend."""
        task = AITask(prompt="hi", model="m1", backend="vllm")
        assert task.backend == "vllm"
        task = AITask(prompt="hi", model="m1", backend="vllm")
        assert task.backend == "vllm"
        assert task["backend"] == "vllm"

    def test_task_default_backend_none(self):
        """Test that tasks have None backend by default."""
        task = ComputeTask(executable="/bin/echo")
        assert task.backend is None
        task = ComputeTask(executable="/bin/echo")
        assert task.backend is None
        assert task["backend"] is None

    def test_optional_fields(self):
        """Test optional fields (input_files, output_files)."""
        task = ComputeTask(
            executable="/bin/app",
            input_files=["/data/input.txt"],
            output_files=["/data/output.txt"],
        )

        assert task.input_files == ["/data/input.txt"]
        assert task.output_files == ["/data/output.txt"]


class TestAITaskValidation:
    """Tests for AITask validation."""

    def test_prompt_with_model_valid(self):
        """Test AITask with prompt and model is valid."""
        task = AITask(prompt="test prompt", model="gpt-4")
        assert task.prompt == "test prompt"
        assert task.model == "gpt-4"

    def test_prompt_with_endpoint_valid(self):
        """Test AITask with prompt and endpoint is valid."""
        task = AITask(prompt="test", inference_endpoint="https://api.example.com")
        assert task.inference_endpoint == "https://api.example.com"

    def test_prompt_required(self):
        """Test prompt is required (TypeError for missing required arg)."""
        with pytest.raises(TypeError):
            # Missing required positional argument 'prompt'
            AITask(model="gpt-4")

    def test_missing_model_and_endpoint_invalid(self):
        """Test missing both model and endpoint raises error."""
        with pytest.raises(TaskValidationError) as exc:
            AITask(prompt="test")

        assert "requires either" in str(exc.value).lower()

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperature
        task = AITask(prompt="test", model="gpt-4", temperature=0.7)
        assert task.temperature == 0.7

        # Invalid temperature (negative)
        with pytest.raises(TaskValidationError):
            AITask(prompt="test", model="gpt-4", temperature=-1.0)

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Valid
        task = AITask(prompt="test", model="gpt-4", max_tokens=1000)
        assert task.max_tokens == 1000

        # Invalid (not positive)
        with pytest.raises(TaskValidationError):
            AITask(prompt="test", model="gpt-4", max_tokens=0)

    def test_top_p_validation(self):
        """Test top_p validation (must be 0-1)."""
        # Valid
        task = AITask(prompt="test", model="gpt-4", top_p=0.9)
        assert task.top_p == 0.9

        # Invalid (> 1)
        with pytest.raises(TaskValidationError):
            AITask(prompt="test", model="gpt-4", top_p=1.5)

        # Invalid (< 0)
        with pytest.raises(TaskValidationError):
            AITask(prompt="test", model="gpt-4", top_p=-0.1)

    def test_top_k_validation(self):
        """Test top_k validation."""
        # Valid
        task = AITask(prompt="test", model="gpt-4", top_k=50)
        assert task.top_k == 50

        # Invalid (not positive)
        with pytest.raises(TaskValidationError):
            AITask(prompt="test", model="gpt-4", top_k=0)

    def test_optional_ai_fields(self):
        """Test optional AI-specific fields."""
        task = AITask(
            prompt="test",
            model="gpt-4",
            system_prompt="You are a helpful assistant",
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9,
            top_k=50,
            stop_sequences=["\n\n", "END"],
        )

        assert task.system_prompt == "You are a helpful assistant"
        assert task.temperature == 0.7
        assert task.max_tokens == 2000
        assert task.top_p == 0.9
        assert task.top_k == 50
        assert task.stop_sequences == ["\n\n", "END"]

    def test_from_dict_compute_task(self):
        """Test from_dict() reconstructs ComputeTask correctly."""
        original = ComputeTask(executable="/bin/echo", arguments=["hello"])

        task_dict = dict(original)
        reconstructed = ComputeTask.from_dict(task_dict)

        assert reconstructed.uid == original.uid
        assert reconstructed.executable == original.executable
        assert reconstructed.arguments == original.arguments

    def test_from_dict_ai_task(self):
        """Test from_dict() reconstructs AITask correctly."""
        original = AITask(prompt="test", model="gpt-4", temperature=0.5, max_tokens=1000)

        task_dict = dict(original)
        reconstructed = AITask.from_dict(task_dict)

        assert reconstructed.uid == original.uid
        assert reconstructed.prompt == original.prompt
        assert reconstructed.model == original.model
        assert reconstructed.temperature == original.temperature
        assert reconstructed.max_tokens == original.max_tokens

    def test_roundtrip_serialization_compute(self):
        """Test ComputeTask → dict → ComputeTask roundtrip."""
        original = ComputeTask(executable="/bin/app", arguments=["arg1"], custom_key="custom_value")

        # Roundtrip
        task_dict = dict(original)
        reconstructed = ComputeTask.from_dict(task_dict)
        final_dict = dict(reconstructed)

        # Should be identical
        assert task_dict == final_dict

    def test_roundtrip_serialization_ai(self):
        """Test AITask → dict → AITask roundtrip."""
        original = AITask(prompt="test", model="gpt-4", temperature=0.7)

        # Roundtrip
        task_dict = dict(original)
        reconstructed = AITask.from_dict(task_dict)
        final_dict = dict(reconstructed)

        # Should be identical
        assert task_dict == final_dict

    def test_base_task_from_dict_auto_detection(self):
        """Test BaseTask.from_dict() auto-detects task type."""
        # ComputeTask dict
        compute_dict = {"uid": "c1", "executable": "/bin/echo"}
        task = rhapsody.BaseTask.from_dict(compute_dict)
        assert isinstance(task, ComputeTask)

        # AITask dict
        ai_dict = {"uid": "a1", "prompt": "test", "model": "gpt-4"}
        task = rhapsody.BaseTask.from_dict(ai_dict)
        assert isinstance(task, AITask)


class TestTaskRepr:
    """Tests for task string representation."""

    def test_compute_task_repr_executable(self):
        """Test ComputeTask __repr__ with executable."""
        task = ComputeTask(uid="task.00000", executable="/bin/echo")

        repr_str = repr(task)

        assert "ComputeTask" in repr_str
        assert "task.00000" in repr_str
        assert "executable" in repr_str

    def test_compute_task_repr_function(self):
        """Test ComputeTask __repr__ with function."""

        def my_func():
            pass

        task = ComputeTask(function=my_func)

        repr_str = repr(task)

        assert "ComputeTask" in repr_str
        assert "function" in repr_str

    def test_ai_task_repr(self):
        """Test AITask __repr__."""
        task = AITask(prompt="test", model="gpt-4")

        repr_str = repr(task)

        assert "AITask" in repr_str
        assert "gpt-4" in repr_str


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_arguments_list(self):
        """Test executable with empty arguments list."""
        task = ComputeTask(executable="/bin/echo", arguments=[])
        assert task.arguments == []

    def test_none_optional_fields(self):
        """Test None values for optional fields."""
        task = ComputeTask(executable="/bin/echo", input_files=None, output_files=None)

        assert task.input_files is None
        assert task.output_files is None

    def test_validate_method(self):
        """Test manual validation via validate() method."""
        task = ComputeTask(executable="/bin/echo")

        # Should not raise
        task.validate()

        # Modify to invalid state and validate
        task["uid"] = ""
        with pytest.raises(TaskValidationError):
            task.validate()

    @pytest.mark.asyncio
    async def test_task_awaitable(self):
        """Test that task can be awaited directly."""
        import asyncio

        task = ComputeTask(executable="/bin/echo")
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        task.bind_future(future)

        # Resolve future
        future.set_result(task)

        # Should be awaitable
        result = await task
        assert result == task

    def test_task_pickling(self):
        """Test that task can be pickled and _future is excluded."""
        import asyncio
        import pickle

        task = ComputeTask(executable="/bin/echo")

        # Create a loop and future (non-pickleable)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        future = loop.create_future()
        task.bind_future(future)

        # Pickle
        data = pickle.dumps(task)

        # Unpickle
        task2 = pickle.loads(data)

        assert task2.uid == task.uid
        assert task2.executable == "/bin/echo"
        assert task2._future is None  # Should be reset to None

    def test_task_internal_attrs(self):
        """Test that _future is in internal attrs and excluded from dictate."""
        task = ComputeTask(executable="/bin/echo")
        assert "_future" in task._INTERNAL_ATTRS

        task_dict = dict(task)
        assert "_future" not in task_dict
