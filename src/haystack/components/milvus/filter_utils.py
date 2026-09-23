# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Filter utilities for Haystack Milvus document store.

Taken from:
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Union

from src.visual_search.exceptions import InputValidationError


def nested_defaultdict() -> defaultdict:
    """
    Data structure that recursively adds a dictionary as value if a key does not exist. Advantage: In nested dictionary
    structures, we don't need to check if a key already exists (which can become hard to maintain in nested dictionaries
    with many levels) but access the existing value if a key exists and create an empty dictionary if a key does not
    exist.
    """
    return defaultdict(nested_defaultdict)


class LogicalFilterClause(ABC):
    """
    Class that is able to parse a filter and convert it to the format that the underlying databases of our
    DocumentStores require.

    Filters are defined as nested dictionaries that can be of two types:
        - Comparison
        - Logic

        Comparison dictionaries must contain the keys:

        - `field`
        - `operator`
        - `value`

        Logic dictionaries must contain the keys:

        - `operator`
        - `conditions`

        The `conditions` key must be a list of dictionaries, either of type Comparison or Logic.

        The `operator` value in Comparison dictionaries must be one of:

        - `==`
        - `!=`
        - `>`
        - `>=`
        - `<`
        - `<=`
        - `in`
        - `not in`

        The `operator` values in Logic dictionaries must be one of:

        - `NOT`
        - `OR`
        - `AND`


        A simple filter:
        ```python
        filters = {"field": "meta.type", "operator": "==", "value": "article"}
        ```

        A more complex filter:
        ```python
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "article"},
                {"field": "meta.date", "operator": ">=", "value": 1420066800},
                {"field": "meta.date", "operator": "<", "value": 1609455600},
                {"field": "meta.rating", "operator": ">=", "value": 3},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.genre", "operator": "in", "value": ["economy", "politics"]},
                        {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
                    ],
                },
            ],
        }
        ```

    """

    def __init__(
        self, conditions: List[Union["LogicalFilterClause", "ComparisonOperation"]]
    ):
        self.conditions = conditions

    @abstractmethod
    def evaluate(self, fields) -> bool:
        pass

    @classmethod
    def parse(
        cls, filters: Dict[str, Any]
    ) -> Union["LogicalFilterClause", "ComparisonOperation"]:
        """
        Parses a filter dictionary/list and returns a LogicalFilterClause instance.

        :param filter_term: Dictionary or list that contains the filter definition.
        """
        conditions: List[Union[LogicalFilterClause, ComparisonOperation]] = []

        if "conditions" in filters:
            operator = filters["operator"]
            for condition in filters["conditions"]:
                if operator == "NOT":
                    conditions.append(NotOperation.parse(condition))
                elif operator == "AND":
                    conditions.append(AndOperation.parse(condition))
                elif operator == "OR":
                    conditions.append(OrOperation.parse(condition))
                else:
                    raise InputValidationError(
                        f"Unknown operator for condition clause: '{operator}'"
                    )
        else:
            required_keys = ["field", "operator", "value"]
            missing_keys = list(filter(lambda key: key not in filters, required_keys))
            if missing_keys:
                raise InputValidationError(
                    f"Missing fields in filter : '{missing_keys}'"
                )
            conditions.extend(
                ComparisonOperation.parse(
                    filters["field"], filters["operator"], filters["value"]
                )
            )

        if cls == LogicalFilterClause:
            if len(conditions) == 1:
                return conditions[0]
            else:
                return AndOperation(conditions)
        else:
            return cls(conditions)

    def convert_to_milvus(self) -> str:
        """
        Converts the LogicalFilterClause instance to a Milvus filter.
        """

    @abstractmethod
    def invert(self) -> Union["LogicalFilterClause", "ComparisonOperation"]:
        """
        Inverts the LogicalOperation instance.
        Necessary for Weaviate as Weaviate doesn't seem to support the 'Not' operator anymore.
        (https://github.com/semi-technologies/weaviate/issues/1717)
        """


class ComparisonOperation(ABC):
    def __init__(
        self, field_name: str, comparison_value: Union[str, int, float, bool, List]
    ):
        self.field_name = field_name
        self.comparison_value = comparison_value

    @abstractmethod
    def evaluate(self, fields) -> bool:
        pass

    @classmethod
    def parse(
        cls, field_name, comparison_operation, comparison_value
    ) -> List["ComparisonOperation"]:
        comparison_operations: List[ComparisonOperation] = []

        if field_name.startswith("meta."):
            # Remove the "meta." prefix if present.
            # Documents are flattened when using the MilvusDocumentStore
            # so we don't need to specify the "meta." prefix.
            field_name = field_name[5:]

        if comparison_operation == "==":
            comparison_operations.append(EqOperation(field_name, comparison_value))
        elif comparison_operation == "in":
            comparison_operations.append(InOperation(field_name, comparison_value))
        elif comparison_operation == "!=":
            comparison_operations.append(NeOperation(field_name, comparison_value))
        elif comparison_operation == "not in":
            comparison_operations.append(NinOperation(field_name, comparison_value))
        elif comparison_operation == ">":
            comparison_operations.append(GtOperation(field_name, comparison_value))
        elif comparison_operation == ">=":
            comparison_operations.append(GteOperation(field_name, comparison_value))
        elif comparison_operation == "<":
            comparison_operations.append(LtOperation(field_name, comparison_value))
        elif comparison_operation == "<=":
            comparison_operations.append(LteOperation(field_name, comparison_value))
        else:
            raise InputValidationError(
                f"unsupported operator in comparison clause: '{comparison_operation}'"
            )

        return comparison_operations

    def convert_to_milvus(self):
        """
        Converts the ComparisonOperation instance to a Milvus comparison operator.
        """

    @abstractmethod
    def invert(self) -> "ComparisonOperation":
        """
        Inverts the ComparisonOperation.
        Necessary for Weaviate as Weaviate doesn't seem to support the 'Not' operator anymore.
        (https://github.com/semi-technologies/weaviate/issues/1717)
        """


class NotOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'NOT' operations.
    """

    def evaluate(self, fields) -> bool:
        return not any(condition.evaluate(fields) for condition in self.conditions)

    def convert_to_milvus(self) -> str:
        conditions = [
            condition.invert().convert_to_milvus() for condition in self.conditions
        ]
        if len(conditions) > 1:
            # Conditions in self.conditions are by default combined with AND which becomes OR according to DeMorgan
            return f"({' or '.join(conditions)})"
        else:
            return conditions[0]

    def invert(self) -> Union[LogicalFilterClause, ComparisonOperation]:
        # This method is called when a "$not" operation is embedded in another "$not" operation. Therefore, we don't
        # invert the operations here, as two "$not" operation annihilate each other.
        # (If we have more than one condition, we return an AndOperation, the default logical operation for combining
        # multiple conditions.)
        if len(self.conditions) > 1:
            return AndOperation(self.conditions)
        else:
            return self.conditions[0]


class AndOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'AND' operations.
    """

    def evaluate(self, fields) -> bool:
        return all(condition.evaluate(fields) for condition in self.conditions)

    def convert_to_milvus(self) -> str:
        conditions = [condition.convert_to_milvus() for condition in self.conditions]
        return f"({' and '.join(conditions)})"

    def invert(self) -> "OrOperation":
        return OrOperation([condition.invert() for condition in self.conditions])


class OrOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'OR' operations.
    """

    def evaluate(self, fields) -> bool:
        return any(condition.evaluate(fields) for condition in self.conditions)

    def convert_to_milvus(self) -> str:
        conditions = [condition.convert_to_milvus() for condition in self.conditions]
        return f"({' or '.join(conditions)})"

    def invert(self) -> AndOperation:
        return AndOperation([condition.invert() for condition in self.conditions])


class EqOperation(ComparisonOperation):
    """
    Handles conversion of the '==' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] == self.comparison_value

    def convert_to_milvus(self) -> str:
        if isinstance(self.comparison_value, str):
            comp_val = '"' + str(self.comparison_value) + '"'
        else:
            comp_val = str(self.comparison_value)
        return f"({self.field_name} == {comp_val})"

    def invert(self) -> "NeOperation":
        return NeOperation(self.field_name, self.comparison_value)


class InOperation(ComparisonOperation):
    """
    Handles conversion of the 'in' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] in self.comparison_value  # type: ignore
        # is only initialized with lists, but changing the type annotation would mean duplicating __init__

    def convert_to_milvus(self) -> str:
        if not isinstance(self.comparison_value, list):
            raise InputValidationError(
                "'in' operation requires comparison value to be a list."
            )
        comp_val = []
        for x in self.comparison_value:
            if isinstance(x, str):
                comp_val.append('"' + str(x) + '"')
            else:
                comp_val.append(str(x))

        return f"""({self.field_name} in [{','.join(comp_val)}])"""

    def invert(self) -> "NinOperation":
        return NinOperation(self.field_name, self.comparison_value)


class NeOperation(ComparisonOperation):
    """
    Handles conversion of the '!=' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] != self.comparison_value

    def convert_to_milvus(self) -> str:
        if isinstance(self.comparison_value, str):
            comp_val = '"' + str(self.comparison_value) + '"'
        else:
            comp_val = str(self.comparison_value)
        return f"({self.field_name} != {comp_val})"

    def invert(self) -> "EqOperation":
        return EqOperation(self.field_name, self.comparison_value)


class NinOperation(ComparisonOperation):
    """
    Handles conversion of the 'not in' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] not in self.comparison_value  # type: ignore
        # is only initialized with lists, but changing the type annotation would mean duplicating __init__

    def convert_to_milvus(self) -> str:
        if not isinstance(self.comparison_value, list):
            raise InputValidationError(
                "'not in' operation requires comparison value to be a list."
            )
        comp_val = []
        for x in self.comparison_value:
            if isinstance(x, str):
                comp_val.append('"' + str(x) + '"')
            else:
                comp_val.append(str(x))

        return f"""({self.field_name} not in [{','.join(comp_val)}])"""

    def invert(self) -> "InOperation":
        return InOperation(self.field_name, self.comparison_value)


class GtOperation(ComparisonOperation):
    """
    Handles conversion of the '>' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] > self.comparison_value

    def convert_to_milvus(self) -> str:
        if not isinstance(self.comparison_value, (float, int, str)):
            raise InputValidationError(
                "Comparison value for '>' operation must be a float or int or str."
            )
        if isinstance(self.comparison_value, str):
            comp_val = '"' + str(self.comparison_value) + '"'
        else:
            comp_val = str(self.comparison_value)
        return f"({self.field_name} > {comp_val})"

    def invert(self) -> "LteOperation":
        return LteOperation(self.field_name, self.comparison_value)


class GteOperation(ComparisonOperation):
    """
    Handles conversion of the '>=' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] >= self.comparison_value

    def convert_to_milvus(self) -> str:
        if not isinstance(self.comparison_value, (float, int, str)):
            raise InputValidationError(
                "Comparison value for '>=' operation must be a float or int or str."
            )
        if isinstance(self.comparison_value, str):
            comp_val = '"' + str(self.comparison_value) + '"'
        else:
            comp_val = str(self.comparison_value)
        return f"({self.field_name} >= {comp_val})"

    def invert(self) -> "LtOperation":
        return LtOperation(self.field_name, self.comparison_value)


class LtOperation(ComparisonOperation):
    """
    Handles conversion of the '<' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] < self.comparison_value

    def convert_to_milvus(self) -> str:
        if not isinstance(self.comparison_value, (float, int, str)):
            raise InputValidationError(
                "Comparison value for '<' operation must be a float or int or str."
            )
        if isinstance(self.comparison_value, str):
            comp_val = '"' + str(self.comparison_value) + '"'
        else:
            comp_val = str(self.comparison_value)
        return f"({self.field_name} < {comp_val})"

    def invert(self) -> "GteOperation":
        return GteOperation(self.field_name, self.comparison_value)


class LteOperation(ComparisonOperation):
    """
    Handles conversion of the '<=' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] <= self.comparison_value

    def convert_to_milvus(self) -> str:
        if not isinstance(self.comparison_value, (float, int, str)):
            raise InputValidationError(
                "Comparison value for '<=' operation must be a float or int or str."
            )
        if isinstance(self.comparison_value, str):
            comp_val = '"' + str(self.comparison_value) + '"'
        else:
            comp_val = str(self.comparison_value)
        return f"({self.field_name} <= {comp_val})"

    def invert(self) -> "GtOperation":
        return GtOperation(self.field_name, self.comparison_value)
