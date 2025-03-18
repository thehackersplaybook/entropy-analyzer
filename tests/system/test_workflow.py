import pytest
from pydantic import BaseModel
from entropy_analyzer.system import BaseWorkflow
from typing import Dict, Optional


class TestInputModel(BaseModel):
    name: str
    age: int
    email: Optional[str] = None


class TestOutputModel(BaseModel):
    message: str
    status: bool


class SuccessWorkflow(BaseWorkflow[TestInputModel, TestOutputModel, Dict]):
    def _run_workflow(self, input_data: TestInputModel) -> Dict:
        return {
            "message": f"Hello {input_data.name}, age {input_data.age}",
            "status": True,
        }


class FailingWorkflow(BaseWorkflow[TestInputModel, TestOutputModel, Dict]):
    def _run_workflow(self, input_data: TestInputModel) -> Dict:
        raise ValueError("Workflow execution failed")


class CustomReturnWorkflow(BaseWorkflow[TestInputModel, TestOutputModel, str]):
    def _run_workflow(self, input_data: TestInputModel) -> str:
        return f"Custom return: {input_data.name}"


@pytest.fixture
def success_workflow():
    return SuccessWorkflow(
        name="Test Workflow",
        description="Test workflow description",
        input_model=TestInputModel,
        output_model=TestOutputModel,
    )


@pytest.fixture
def failing_workflow():
    return FailingWorkflow(
        name="Failing Workflow",
        description="Workflow that fails",
        input_model=TestInputModel,
        output_model=TestOutputModel,
    )


@pytest.fixture
def verbose_workflow():
    return SuccessWorkflow(
        name="Verbose Workflow",
        description="Workflow with verbose logging",
        input_model=TestInputModel,
        output_model=TestOutputModel,
        verbose=True,
    )


def test_successful_workflow_execution(success_workflow):
    input_data = {"name": "John", "age": 30}
    result = success_workflow.run(input_data)
    assert isinstance(result, TestOutputModel)
    assert result.message == "Hello John, age 30"
    assert result.status is True


def test_workflow_with_invalid_input(success_workflow):
    invalid_input = {"name": "John"}
    with pytest.raises(SystemError) as exc_info:
        success_workflow.run(invalid_input)
    assert "Invalid input data" in str(exc_info.value)


def test_workflow_with_model_instance_input(success_workflow):
    input_model = TestInputModel(name="Jane", age=25)
    result = success_workflow.run(input_model)
    assert isinstance(result, TestOutputModel)
    assert result.message == "Hello Jane, age 25"


def test_workflow_execution_failure(failing_workflow):
    input_data = {"name": "John", "age": 30}
    with pytest.raises(SystemError) as exc_info:
        failing_workflow.run(input_data)
    assert "Workflow execution failed" in str(exc_info.value)


def test_workflow_with_custom_id():
    custom_id = "test-workflow-123"
    workflow = SuccessWorkflow(
        workflow_id=custom_id, input_model=TestInputModel, output_model=TestOutputModel
    )
    assert workflow.workflow_id == custom_id


def test_workflow_without_models():
    class TestBaseWorkflow(BaseWorkflow):
        def _run_workflow(self, input_data):
            return input_data

    workflow = TestBaseWorkflow()
    input_data = {"test": "data"}
    result = workflow.run(input_data)
    assert result == input_data


def test_workflow_with_optional_fields():
    workflow = SuccessWorkflow(input_model=TestInputModel, output_model=TestOutputModel)
    input_data = {"name": "John", "age": 30, "email": "john@example.com"}
    result = workflow.run(input_data)
    assert isinstance(result, TestOutputModel)


def test_workflow_verbose_logging(verbose_workflow, capfd):
    input_data = {"name": "John", "age": 30}
    verbose_workflow.run(input_data)
    captured = capfd.readouterr()
    assert "Starting workflow" in captured.out
    assert "Input data" in captured.out
    assert "Workflow completed successfully" in captured.out


def test_custom_return_type():
    class NoOutputModelWorkflow(BaseWorkflow[TestInputModel, None, str]):
        def _run_workflow(self, input_data: TestInputModel) -> str:
            return f"Custom return: {input_data.name}"

    workflow = NoOutputModelWorkflow(input_model=TestInputModel)
    result = workflow.run({"name": "John", "age": 30})
    assert isinstance(result, str)
    assert result == "Custom return: John"


def test_workflow_initialization_error():
    with pytest.raises(RuntimeError) as exc_info:
        SuccessWorkflow(
            workflow_id=123,
            input_model=TestInputModel,
            output_model=TestOutputModel,
        )
    assert "Invalid workflow ID type" in str(exc_info.value)
