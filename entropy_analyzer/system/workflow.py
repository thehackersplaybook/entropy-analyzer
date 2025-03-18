"""
Workflow System Core Module.

This module provides the foundation for creating and executing workflows in the system.
It implements a generic workflow pattern that supports input validation, output transformation,
and comprehensive logging capabilities.

The workflow system is designed to be:
- Type-safe: Using generic types for input/output validation.
- Extensible: Easy to create new workflow implementations.
- Traceable: Comprehensive logging of workflow execution.
- Error-resistant: Built-in error handling and logging.

The module uses Pydantic for data validation and transformation, making it
robust for handling complex data structures while maintaining type safety.

Author: Aditya Patange (AdiPat)
License: MIT
"""

import traceback
import uuid
from typing import Any, Dict, Optional, TypeVar, Generic, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel
from .printer import Printer


I = TypeVar("I", bound=BaseModel)
T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")


class BaseWorkflow(ABC, Generic[I, T, R]):
    """Base abstract class for creating workflow implementations.

    This class serves as the foundation for all workflow implementations in the system.
    It provides a structured way to process data through a workflow with input validation,
    logging, and error handling built-in. Each concrete implementation must provide
    the core workflow logic through the _run_workflow method.

    The class uses generics to ensure type safety:
    - I: Input model type (must be a Pydantic BaseModel).
    - T: Output model type (must be a Pydantic BaseModel).
    - R: Custom return type for the workflow.

    Generic workflows can be created by extending this class and implementing the
    required _run_workflow method.
    """

    def __init__(
        self,
        workflow_id: str = None,
        name: str = "",
        description: str = "",
        input_model: Optional[type[I]] = None,
        output_model: Optional[type[T]] = None,
        verbose: bool = False,
    ):
        """Initialize a new workflow instance.

        Args:
            workflow_id: Unique identifier for the workflow. If not provided,
                       a new UUID will be generated.
            name: Human-readable name for the workflow. Used in logging and
                 identification.
            description: Detailed description of what the workflow does.
            input_model: Optional Pydantic model class that defines the expected
                        input data structure.
            output_model: Optional Pydantic model class that defines the expected
                         output data structure.
            verbose: When True, enables detailed debug logging for the workflow
                    execution.

        Raises:
            RuntimeError: If initialization fails due to invalid arguments.
        """
        try:
            if workflow_id is not None and not isinstance(workflow_id, str):
                raise ValueError(
                    f"Invalid workflow ID type [{workflow_id}]. Only strings allowed."
                )
            self.workflow_id = workflow_id or str(uuid.uuid4())
            self.name = name
            self.description = description
            self.input_model = input_model
            self.output_model = output_model
            self.verbose = verbose

        except Exception as e:
            Printer.print_red_message(f"Failed to initialize workflow: {str(e)}.")
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize workflow: {str(e)}.") from e

    def run(self, input_data: Union[I, Dict, Any]) -> Union[Dict, T, R]:
        """Execute the workflow with the given input data.

        This method handles the complete workflow execution lifecycle including:
        - Input validation and transformation.
        - Workflow execution.
        - Output validation and transformation.
        - Error handling and logging.

        Args:
            input_data: Input data that matches input_model if specified.

        Returns:
            Union[Dict, T, R]: Either a Dict, the specified output_model T, or custom return type R.

        Raises:
            ValueError: If input data validation fails.
            RuntimeError: If workflow execution fails.
            Exception: Any other unexpected errors during execution.
            SystemError: If workflow fails due to an unexpected error.
        """
        try:
            if self.input_model:
                try:
                    input_data = (
                        self.input_model(**input_data)
                        if not isinstance(input_data, self.input_model)
                        else input_data
                    )
                except Exception as e:
                    Printer.print_red_message(f"Input validation failed: {str(e)}.")
                    Printer.print_red_message(f"Input data: {input_data}.")
                    traceback.print_exc()
                    raise ValueError(f"Invalid input data: {str(e)}.") from e

            Printer.verbose_logger(
                self.verbose,
                Printer.print_light_grey_message,
                f"Starting workflow: {self.name} ({self.workflow_id}).",
            )

            Printer.verbose_logger(
                self.verbose,
                Printer.print_light_grey_message,
                f"Input data: {input_data}.",
            )

            Printer.verbose_logger(
                self.verbose,
                Printer.print_light_grey_message,
                f"Description: {self.description}.",
            )

            try:
                result = self._run_workflow(input_data)
            except Exception as e:
                Printer.print_red_message(f"Workflow execution failed: {str(e)}.")
                traceback.print_exc()
                raise RuntimeError(f"Workflow execution failed: {str(e)}.") from e

            if self.output_model:
                try:
                    result = (
                        self.output_model(**result)
                        if not isinstance(result, self.output_model)
                        else result
                    )
                except Exception as e:
                    Printer.print_red_message(f"Output validation failed: {str(e)}.")
                    Printer.print_red_message(f"Output data: {result}.")
                    traceback.print_exc()
                    raise ValueError(f"Invalid output data: {str(e)}.") from e

            Printer.verbose_logger(
                self.verbose,
                Printer.print_light_grey_message,
                f"Workflow result: {result}.",
            )

            Printer.verbose_logger(
                self.verbose,
                Printer.print_light_grey_message,
                f"Workflow completed successfully: {self.name}.",
            )

            return result

        except Exception as e:
            Printer.print_red_message(f"Workflow failed: {str(e)}.")
            traceback.print_exc()
            raise SystemError(f"Workflow failed: {str(e)}.") from e

    @abstractmethod
    def _run_workflow(self, input_data: I) -> Union[Union[R, None], str]:
        """Abstract method that must be implemented by child classes to define workflow logic.

        Args:
            input_data: Input data of type I (BaseModel).
        Returns:
            Union[Union[R, None], str]: The workflow result of type R or None if no result is expected and the error message as a string.
        """
        pass
