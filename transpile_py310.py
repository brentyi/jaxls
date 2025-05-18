#!/usr/bin/env python3
"""Transpile Python source files to Python 3.10 compatibility.

Removes type annotations (replacing with Any), comments, and docstrings.
Preserves dataclass compatibility and handles generic classes properly.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import libcst as cst
import tyro


class PythonTranspiler(cst.CSTTransformer):
    """Transform Python code for Python 3.10 compatibility."""
    
    def __init__(self):
        self.needs_any_import = False
        self.has_any_import = False
        self.overloaded_functions = set()
    
    # ==================== Type Annotation Handling ====================
    
    def leave_Annotation(self, original_node, updated_node):
        """Replace type annotations with Any, preserving wrapper types."""
        if isinstance(updated_node.annotation, cst.Subscript):
            # Handle ClassVar[T], jdc.Static[T], etc.
            if self._is_special_wrapper(updated_node.annotation):
                return self._replace_inner_type_with_any(updated_node)
        
        # Replace all other annotations with Any
        self.needs_any_import = True
        return updated_node.with_changes(annotation=cst.Name("Any"))
    
    def leave_AnnAssign(self, original_node, updated_node):
        """Replace type annotations in assignments, handling special wrappers."""
        if updated_node.annotation:
            annotation = updated_node.annotation.annotation
            if isinstance(annotation, cst.Subscript) and self._is_special_wrapper(annotation):
                new_annotation = self._create_wrapper_with_any(annotation)
                return updated_node.with_changes(annotation=new_annotation)
        
        # Replace with Any for all other cases
        self.needs_any_import = True
        return updated_node.with_changes(
            annotation=cst.Annotation(annotation=cst.Name("Any"))
        )
    
    def _is_special_wrapper(self, subscript):
        """Check if subscript is a special wrapper like ClassVar or jdc.Static."""
        if isinstance(subscript.value, cst.Name):
            return subscript.value.value == "ClassVar"
        elif isinstance(subscript.value, cst.Attribute):
            attr = subscript.value
            return (isinstance(attr.attr, cst.Name) and 
                   attr.attr.value in ["Static", "ClassVar"])
        return False
    
    def _replace_inner_type_with_any(self, node):
        """Replace inner type of wrapper with Any."""
        self.needs_any_import = True
        return node.with_changes(
            annotation=node.annotation.with_changes(
                slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Name("Any")))]
            )
        )
    
    def _create_wrapper_with_any(self, subscript):
        """Create a wrapper annotation with Any as inner type."""
        self.needs_any_import = True
        return cst.Annotation(
            annotation=subscript.with_changes(
                slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Name("Any")))]
            )
        )
    
    # ==================== Import Management ====================
    
    def leave_ImportFrom(self, original_node, updated_node):
        """Manage typing imports, keeping only necessary ones."""
        if not self._is_typing_import(original_node):
            return updated_node
        
        # Track if Any is imported
        self._check_for_any_import(original_node)
        
        # Filter typing imports
        if hasattr(original_node, 'names') and isinstance(original_node.names, cst.ImportStar):
            return updated_node
        
        if original_node.names:
            new_names = self._filter_typing_imports(original_node.names)
            if new_names:
                return self._fix_import_commas(updated_node, new_names)
            else:
                return cst.RemovalSentinel.REMOVE
        
        return updated_node
    
    def _is_typing_import(self, node):
        """Check if import is from typing modules."""
        return (node.module and 
                node.module.value in ("typing", "typing_extensions", "collections.abc"))
    
    def _check_for_any_import(self, node):
        """Check if Any is being imported."""
        if node.names and not isinstance(node.names, cst.ImportStar):
            for name in node.names:
                if isinstance(name, cst.ImportAlias) and name.name.value == "Any":
                    self.has_any_import = True
                    break
    
    def _filter_typing_imports(self, names):
        """Filter out type-only imports while keeping runtime types."""
        # Types to remove (used only for type checking)
        typing_only = {
            "TYPE_CHECKING", "Union", "List", "Dict", "Set", "Tuple", "Type",
            "TypeVar", "Generic", "Protocol", "Literal", "Final", "Awaitable",
            "Iterator", "Sequence", "Mapping", "Counter", "Deque", "ChainMap",
            "cast", "overload", "NewType", "NamedTuple", "TypedDict",
            "get_type_hints", "get_origin", "get_args", "Annotated", "TypeAlias",
            "TypeGuard", "NoReturn", "Never", "Self", "Unpack", "TypeVarTuple",
            "ParamSpec", "Concatenate", "assert_never", "Optional", "NotRequired",
            "Required", "abstractmethod", "ABC",
        }
        
        # Types to keep (used at runtime)
        runtime_types = {"Any", "deprecated", "ClassVar", "Callable", "Hashable", "Iterable"}
        
        new_names = []
        for name in names:
            if isinstance(name, cst.ImportAlias):
                if name.name.value in runtime_types or name.name.value not in typing_only:
                    new_names.append(name)
        
        return new_names
    
    def _fix_import_commas(self, node, names):
        """Fix trailing commas in import statements."""
        if len(names) == 1:
            names[0] = names[0].with_changes(comma=cst.MaybeSentinel.DEFAULT)
        else:
            names[-1] = names[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
        return node.with_changes(names=names)
    
    # ==================== Comment and Docstring Removal ====================
    
    def leave_TrailingWhitespace(self, original_node, updated_node):
        """Remove comments from trailing whitespace."""
        if updated_node.comment:
            return updated_node.with_changes(comment=None)
        return updated_node
    
    def leave_Comment(self, original_node, updated_node):
        """Remove standalone comments."""
        return cst.RemovalSentinel.REMOVE
    
    def leave_SimpleStatementLine(self, original_node, updated_node):
        """Remove isolated string literals (usually field documentation)."""
        if self._is_docstring_statement(updated_node):
            return cst.RemovalSentinel.REMOVE
        
        # Remove trailing comments from statements
        if updated_node.trailing_whitespace and updated_node.trailing_whitespace.comment:
            new_whitespace = updated_node.trailing_whitespace.with_changes(comment=None)
            return updated_node.with_changes(trailing_whitespace=new_whitespace)
            
        return updated_node
    
    def leave_EmptyLine(self, original_node, updated_node):
        """Remove empty lines with only comments."""
        if updated_node.comment:
            return cst.RemovalSentinel.REMOVE
        return updated_node
    
    def leave_Module(self, original_node, updated_node):
        """Remove module docstring."""
        if updated_node.body:
            new_body = self._remove_docstring(updated_node.body)
            return updated_node.with_changes(body=new_body)
        return updated_node
    
    def _is_docstring_statement(self, node):
        """Check if statement is just a string literal."""
        return (len(node.body) == 1 and 
                isinstance(node.body[0], cst.Expr) and
                isinstance(node.body[0].value, (cst.SimpleString, cst.ConcatenatedString)))
    
    def _remove_docstring(self, body_list):
        """Remove first string literal from a body (docstring)."""
        if not body_list:
            return body_list
        
        new_body = []
        for i, stmt in enumerate(body_list):
            # Skip the first string literal expression
            if i == 0 and isinstance(stmt, cst.SimpleStatementLine):
                if self._is_docstring_statement(stmt):
                    continue
            new_body.append(stmt)
        
        return new_body
    
    # ==================== Function Handling ====================
    
    def leave_FunctionDef(self, original_node, updated_node):
        """Handle function definitions, removing overloads and type parameters."""
        # Remove @overload functions
        if self._has_overload_decorator(original_node):
            self.overloaded_functions.add(original_node.name.value)
            return cst.RemovalSentinel.REMOVE
        
        # Remove overload decorators and type parameters
        updated_node = self._clean_function_decorators(updated_node)
        
        # Remove PEP 695 type parameters
        if hasattr(updated_node, 'type_parameters') and updated_node.type_parameters:
            updated_node = updated_node.with_changes(type_parameters=None)
        
        # Remove docstring
        if isinstance(updated_node.body, cst.IndentedBlock):
            new_body = self._remove_docstring(updated_node.body.body)
            updated_node = updated_node.with_changes(
                body=updated_node.body.with_changes(body=new_body)
            )
        
        return updated_node
    
    def _has_overload_decorator(self, node):
        """Check if function has @overload decorator."""
        return any(
            isinstance(d.decorator, cst.Name) and d.decorator.value == "overload"
            for d in node.decorators
        )
    
    def _clean_function_decorators(self, node):
        """Remove overload decorators from function."""
        new_decorators = [
            d for d in node.decorators
            if not (isinstance(d.decorator, cst.Name) and d.decorator.value == "overload")
        ]
        return node.with_changes(decorators=new_decorators)
    
    # ==================== Class Handling ====================
    
    def leave_ClassDef(self, original_node, updated_node):
        """Handle generic classes, removing type parameters and adding __class_getitem__."""
        needs_class_getitem = False
        
        # Remove PEP 695 type parameters
        if hasattr(updated_node, 'type_parameters') and updated_node.type_parameters:
            updated_node = updated_node.with_changes(type_parameters=None)
            needs_class_getitem = True
        
        # Handle generic base classes
        if updated_node.bases:
            new_bases, generic_found = self._process_class_bases(updated_node.bases)
            updated_node = updated_node.with_changes(bases=new_bases)
            needs_class_getitem = needs_class_getitem or generic_found
        
        # Remove docstring
        new_body = []
        if isinstance(updated_node.body, cst.IndentedBlock):
            new_body = self._remove_docstring(updated_node.body.body)
            updated_node = updated_node.with_changes(
                body=updated_node.body.with_changes(body=new_body)
            )
        
        # Add __class_getitem__ if needed
        if needs_class_getitem and not self._has_class_getitem(updated_node):
            class_getitem = self._create_class_getitem_method()
            if isinstance(updated_node.body, cst.SimpleStatementSuite):
                # Convert SimpleStatementSuite body to proper statements
                statements = [
                    cst.SimpleStatementLine(body=[stmt]) 
                    if isinstance(stmt, cst.BaseSmallStatement) 
                    else stmt 
                    for stmt in updated_node.body.body
                ]
                updated_node = updated_node.with_changes(
                    body=cst.IndentedBlock(body=[class_getitem] + statements)
                )
            else:
                new_body = [class_getitem, *new_body]
                updated_node = updated_node.with_changes(
                    body=updated_node.body.with_changes(body=new_body)
                )
        
        return updated_node
    
    def _process_class_bases(self, bases):
        """Process class bases, handling generics."""
        new_bases = []
        generic_found = False
        
        for base in bases:
            if isinstance(base.value, cst.Subscript):
                if isinstance(base.value.value, cst.Name):
                    if base.value.value.value == "Generic":
                        generic_found = True
                        continue  # Skip Generic base
                    else:
                        # Replace Var[T] with Var[Any]
                        self.needs_any_import = True
                        generic_found = True
                        subscripted_base = base.value.with_changes(
                            slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Name("Any")))]
                        )
                        new_bases.append(cst.Arg(subscripted_base))
                else:
                    new_bases.append(base)
            else:
                new_bases.append(base)
        
        return new_bases, generic_found
    
    def _has_class_getitem(self, class_node):
        """Check if class already has __class_getitem__ method."""
        for item in class_node.body.body:
            if isinstance(item, cst.FunctionDef) and item.name.value == "__class_getitem__":
                return True
        return False
    
    def _create_class_getitem_method(self):
        """Create __class_getitem__ method for generic classes."""
        return cst.FunctionDef(
            name=cst.Name("__class_getitem__"),
            params=cst.Parameters(params=[
                cst.Param(cst.Name("cls")),
                cst.Param(cst.Name("params"))
            ]),
            body=cst.IndentedBlock(body=[
                cst.SimpleStatementLine(body=[cst.Return(cst.Name("cls"))])
            ]),
            decorators=[cst.Decorator(cst.Name("classmethod"))],
        )
    
    # ==================== Other Transformations ====================
    
    def leave_Lambda(self, original_node, updated_node):
        """Remove type annotations from lambda parameters."""
        new_params = updated_node.params.with_changes(
            params=[p.with_changes(annotation=None) for p in updated_node.params.params],
        )
        return updated_node.with_changes(params=new_params)
    
    def leave_Subscript(self, original_node, updated_node):
        """Handle generic type subscripts."""
        # Skip array slicing (not type annotations)
        if any(isinstance(elem.slice, cst.Slice) for elem in updated_node.slice):
            return updated_node
        
        # Special handling for ClassVar[T]
        if (isinstance(updated_node.value, cst.Name) and 
            updated_node.value.value == "ClassVar"):
            self.needs_any_import = True
            return updated_node.with_changes(
                slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Name("Any")))]
            )
        
        # Replace typing constructs with Any
        if isinstance(updated_node.value, cst.Name):
            typing_constructs = {
                "List", "Dict", "Set", "Tuple", "Optional", "Union",
                "Callable", "Type", "Iterable", "Iterator", "Sequence",
                "Mapping", "Generic", "TypeVar", "Literal", "Annotated",
                "TypeAlias", "ParamSpec", "TypeVarTuple", "Unpack"
            }
            if updated_node.value.value in typing_constructs:
                self.needs_any_import = True
                return cst.Name("Any")
        
        return updated_node
    
    def leave_If(self, original_node, updated_node):
        """Handle TYPE_CHECKING blocks."""
        if (isinstance(updated_node.test, cst.Name) and 
            updated_node.test.value == "TYPE_CHECKING"):
            return cst.RemovalSentinel.REMOVE
        
        # Handle if not TYPE_CHECKING
        elif isinstance(updated_node.test, cst.UnaryOperation):
            if (isinstance(updated_node.test.operator, cst.Not) and
                isinstance(updated_node.test.expression, cst.Name) and
                updated_node.test.expression.value == "TYPE_CHECKING"):
                return updated_node.with_changes(test=cst.Name("True"))
        
        return updated_node
    
    def leave_Call(self, original_node, updated_node):
        """Handle cast() calls."""
        if isinstance(updated_node.func, cst.Name):
            if updated_node.func.value == "cast":
                # cast(Type, value) -> value
                if len(updated_node.args) >= 2:
                    return updated_node.args[1].value
        
        return updated_node
    


def transpile_file(input_path: Path, output_path: Path) -> None:
    """Transpile a single Python file."""
    try:
        source_code = input_path.read_text()
        
        # Skip empty files
        if not source_code.strip():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(source_code)
            return
        
        # Parse and transform
        tree = cst.parse_module(source_code)
        transformer = PythonTranspiler()
        modified_tree = tree.visit(transformer)
        
        # Add Any import if needed
        if transformer.needs_any_import and not transformer.has_any_import:
            modified_tree = add_any_import(modified_tree)
        
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(modified_tree.code)
        print(f"Processed: {input_path} -> {output_path}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        # Copy file as-is on error
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)


def add_any_import(tree):
    """Add 'from typing import Any' to module."""
    # Find position after __future__ imports
    insert_position = 0
    for i, stmt in enumerate(tree.body):
        if isinstance(stmt, cst.SimpleStatementLine):
            for expr in stmt.body:
                if isinstance(expr, cst.ImportFrom):
                    if expr.module and expr.module.value == "__future__":
                        insert_position = i + 1
        elif not isinstance(stmt, (cst.SimpleStatementLine, cst.EmptyLine)):
            break
    
    # Create import statement
    import_any = cst.SimpleStatementLine(
        body=[cst.ImportFrom(
            module=cst.Name("typing"),
            names=[cst.ImportAlias(name=cst.Name("Any"))]
        )]
    )
    
    # Insert import
    new_body = list(tree.body)
    new_body.insert(insert_position, import_any)
    return tree.with_changes(body=new_body)


def transpile_directory(input_dir: Path, output_dir: Path) -> None:
    """Transpile all Python files in directory."""
    python_files = list(input_dir.glob("**/*.py"))
    print(f"Found {len(python_files)} Python files to process")
    
    for py_file in python_files:
        relative_path = py_file.relative_to(input_dir)
        output_path = output_dir / relative_path
        transpile_file(py_file, output_path)
    
    # Copy non-Python files
    non_python_files = []
    for file_path in input_dir.glob("**/*"):
        if file_path.is_file() and not file_path.suffix == ".py":
            non_python_files.append(file_path)
    
    print(f"Found {len(non_python_files)} non-Python files to copy")
    
    for file_path in non_python_files:
        relative_path = file_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, output_path)
        print(f"Copied: {file_path} -> {output_path}")


def main(
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Transpile Python source files to Python 3.10 compatibility.
    
    Args:
        input_dir: Directory containing Python files to process.
        output_dir: Directory where transpiled files will be written.
    """
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        return
    
    print(f"Transpiling {input_dir} to {output_dir}")
    transpile_directory(input_dir, output_dir)
    print("Done!")


if __name__ == "__main__":
    tyro.cli(main)