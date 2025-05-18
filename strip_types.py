#!/usr/bin/env python3
"""Strip type annotations from Python source files."""

from __future__ import annotations

import shutil
from pathlib import Path

import libcst as cst
import tyro


class TypeStripper(cst.CSTTransformer):
    """Replace type annotations with Any."""
    
    def __init__(self):
        self.needs_any_import = False
        self.has_any_import = False
        # Track overloaded functions by name
        self.overloaded_functions = set()

    def leave_AnnAssign(self, original_node, updated_node):
        """Replace type annotations with Any in annotated assignments."""
        # Special case: if the annotation contains ClassVar or jdc.Static, keep it but replace the inner type
        if updated_node.annotation and isinstance(updated_node.annotation.annotation, cst.Subscript):
            subscript = updated_node.annotation.annotation
            if isinstance(subscript.value, cst.Name) and subscript.value.value == "ClassVar":
                self.needs_any_import = True
                # Replace ClassVar[T] with ClassVar[Any]
                new_annotation = cst.Annotation(
                    annotation=subscript.with_changes(
                        slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Name("Any")))]
                    )
                )
                return updated_node.with_changes(annotation=new_annotation)
            elif isinstance(subscript.value, cst.Attribute):
                # Check for jdc.Static[T], typing.ClassVar[T], etc.
                attr = subscript.value
                # Debug print
                # print(f"AnnAssign: found attribute {attr.attr.value if isinstance(attr.attr, cst.Name) else 'unknown'}")
                if (isinstance(attr.attr, cst.Name) and 
                    attr.attr.value in ["Static", "ClassVar"]):
                    self.needs_any_import = True
                    # Replace Mod.Attr[T] with Mod.Attr[Any]
                    new_annotation = cst.Annotation(
                        annotation=subscript.with_changes(
                            slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Name("Any")))]
                        )
                    )
                    return updated_node.with_changes(annotation=new_annotation)
        
        # Otherwise, replace with Any
        self.needs_any_import = True
        if updated_node.value is None:
            # x: int -> x: Any
            return updated_node.with_changes(annotation=cst.Annotation(annotation=cst.Name("Any")))
        else:
            # x: int = 5 -> x: Any = 5
            return updated_node.with_changes(annotation=cst.Annotation(annotation=cst.Name("Any")))
    
    def leave_Assign(self, original_node, updated_node):
        """Handle type alias assignments."""
        # Check if this looks like a type alias assignment
        if len(updated_node.targets) == 1:
            target = updated_node.targets[0].target
            if isinstance(target, cst.Name) and isinstance(updated_node.value, cst.Name):
                # Handle cases like: CustomJacobianCache = Any
                if updated_node.value.value == "Any":
                    self.needs_any_import = True
        
        return updated_node
    
    def leave_Annotation(self, original_node, updated_node):
        """Replace type annotations with Any."""
        # Special case: if the annotation contains ClassVar or jdc.Static, keep it but replace the inner type
        if isinstance(updated_node.annotation, cst.Subscript):
            # Handle ClassVar[T]
            if isinstance(updated_node.annotation.value, cst.Name) and updated_node.annotation.value.value == "ClassVar":
                self.needs_any_import = True
                # Replace ClassVar[T] with ClassVar[Any]
                return updated_node.with_changes(
                    annotation=updated_node.annotation.with_changes(
                        slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Name("Any")))]
                    )
                )
            # Handle jdc.Static[T] or typing.ClassVar[T]
            elif isinstance(updated_node.annotation.value, cst.Attribute):
                attr = updated_node.annotation.value
                if isinstance(attr.attr, cst.Name) and attr.attr.value in ["Static", "ClassVar"]:
                    self.needs_any_import = True
                    # Replace Mod.Attr[T] with Mod.Attr[Any]
                    return updated_node.with_changes(
                        annotation=updated_node.annotation.with_changes(
                            slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Name("Any")))]
                        )
                    )
        
        # Otherwise, replace with Any
        self.needs_any_import = True
        return updated_node.with_changes(annotation=cst.Name("Any"))

    def leave_FunctionDef(self, original_node, updated_node):
        """Handle function definitions."""
        # Check if this function has @overload decorator
        has_overload = any(
            isinstance(d.decorator, cst.Name) and d.decorator.value == "overload"
            for d in original_node.decorators
        )
        
        if has_overload:
            # Mark this function as overloaded and remove it entirely
            self.overloaded_functions.add(original_node.name.value)
            return cst.RemovalSentinel.REMOVE
        
        # If this function name was previously seen with @overload, it's the implementation
        # Keep it but remove any overload-related decorators
        new_decorators = [
            d for d in updated_node.decorators
            if not (isinstance(d.decorator, cst.Name) and d.decorator.value == "overload")
        ]
        
        # Also handle PEP 695 style type parameters if present
        if hasattr(updated_node, 'type_parameters') and updated_node.type_parameters:
            updated_node = updated_node.with_changes(type_parameters=None)
        
        return updated_node.with_changes(decorators=new_decorators)

    def leave_ClassDef(self, original_node, updated_node):
        """Handle generic classes and class type parameters."""
        # Track if this class needs __class_getitem__
        needs_class_getitem = False
        
        # Handle PEP 695 generic syntax (Python 3.12+)
        if hasattr(updated_node, 'type_parameters') and updated_node.type_parameters:
            # Remove type parameters from class
            updated_node = updated_node.with_changes(type_parameters=None)
            needs_class_getitem = True
        
        # Handle bases with generics
        if updated_node.bases:
            new_bases = []
            for base in updated_node.bases:
                if isinstance(base.value, cst.Subscript):
                    # If base is subscripted (like Generic[T] or Cost[*Args])
                    if isinstance(base.value.value, cst.Name):
                        if base.value.value.value == "Generic":
                            # Skip Generic base entirely
                            needs_class_getitem = True
                            continue
                        else:
                            # Keep the base class but replace subscript with Any
                            # Transform Var[jaxlie.SO2] -> Var[Any]
                            self.needs_any_import = True
                            needs_class_getitem = True  # This class also needs __class_getitem__
                            subscripted_base = base.value.with_changes(
                                slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Name("Any")))]
                            )
                            new_bases.append(cst.Arg(subscripted_base))
                    else:
                        # For more complex base expressions, keep the base without subscript
                        new_bases.append(base)
                else:
                    new_bases.append(base)
            
            # Update the bases if we made changes - but keep keywords intact!
            updated_node = updated_node.with_changes(bases=new_bases)
        
        # Add __class_getitem__ method if the class was generic
        if needs_class_getitem and not self._has_class_getitem(updated_node):
            # Create __class_getitem__ method that returns cls
            class_getitem = cst.FunctionDef(
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
            
            # Insert at the beginning of the class body
            if isinstance(updated_node.body, cst.SimpleStatementSuite):
                # Need to convert SimpleStatementSuite to IndentedBlock
                updated_node = updated_node.with_changes(
                    body=cst.IndentedBlock(body=[
                        class_getitem,
                        *updated_node.body.body
                    ])
                )
            else:
                new_body = [class_getitem, *updated_node.body.body]
                updated_node = updated_node.with_changes(
                    body=updated_node.body.with_changes(body=new_body)
                )
        
        return updated_node

    def _has_class_getitem(self, class_node):
        """Check if the class already has a __class_getitem__ method."""
        for item in class_node.body.body:
            if isinstance(item, cst.FunctionDef) and item.name.value == "__class_getitem__":
                return True
        return False

    def leave_ImportFrom(self, original_node, updated_node):
        """Track and modify typing imports."""
        if original_node.module and original_node.module.value in (
            "typing", "typing_extensions", "collections.abc"
        ):
            # Check if Any is being imported
            if original_node.names and not isinstance(original_node.names, cst.ImportStar):
                for name in original_node.names:
                    if isinstance(name, cst.ImportAlias) and name.name.value == "Any":
                        self.has_any_import = True
                        break
            
            # Check if it's for TYPE_CHECKING only
            if hasattr(original_node, 'names') and isinstance(original_node.names, cst.ImportStar):
                return updated_node  # Keep import *
            
            # Filter out typing-only names but keep Any, deprecated, and ClassVar
            typing_only = {
                # From typing module
                "TYPE_CHECKING", "Union", "List", "Dict",
                "Set", "Tuple", "Type", "TypeVar", "Generic", "Protocol",
                "Literal", "Final", "Callable", "Awaitable", "Iterable",
                "Iterator", "Sequence", "Mapping", "Counter", "Deque",
                "ChainMap", "cast", "overload", "NewType",
                "NamedTuple", "TypedDict", "get_type_hints", "get_origin",
                "get_args", "Annotated", "TypeAlias", "TypeGuard",
                "NoReturn", "Never", "Self", "Unpack", "TypeVarTuple",
                "ParamSpec", "Concatenate", "assert_never", "Optional",
                # From typing_extensions (excluding deprecated)
                "NotRequired", "Required", "TypedDict",
                # Other typing-related
                "abstractmethod", "ABC",
            }
            
            if original_node.names:
                new_names = []
                for name in original_node.names:
                    if isinstance(name, cst.ImportAlias):
                        # Keep Any, deprecated, ClassVar, and types used at runtime
                        if name.name.value in ("Any", "deprecated", "ClassVar", "Callable", "Hashable", "Iterable"):
                            new_names.append(name)
                        elif name.name.value not in typing_only:
                            new_names.append(name)
                
                if new_names:
                    # Fix trailing commas in import statement
                    if len(new_names) == 1:
                        # Single import should not have trailing comma
                        new_names[0] = new_names[0].with_changes(comma=cst.MaybeSentinel.DEFAULT)
                    else:
                        # Make sure the last import doesn't have a trailing comma
                        new_names[-1] = new_names[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
                    return updated_node.with_changes(names=new_names)
                else:
                    # Remove the entire import statement
                    return cst.RemovalSentinel.REMOVE
        
        return updated_node

    def leave_If(self, original_node, updated_node):
        """Handle TYPE_CHECKING blocks."""
        # Check for if TYPE_CHECKING: block
        if (isinstance(updated_node.test, cst.Name) and 
            updated_node.test.value == "TYPE_CHECKING"):
            # Remove the entire if block
            return cst.RemovalSentinel.REMOVE
        
        # Check for if not TYPE_CHECKING: block
        elif isinstance(updated_node.test, cst.UnaryOperation):
            if (isinstance(updated_node.test.operator, cst.Not) and
                isinstance(updated_node.test.expression, cst.Name) and
                updated_node.test.expression.value == "TYPE_CHECKING"):
                # Replace `if not TYPE_CHECKING:` with `if True:`
                return updated_node.with_changes(test=cst.Name("True"))
        
        return updated_node

    def leave_Call(self, original_node, updated_node):
        """Handle cast() and assert_never() calls."""
        if isinstance(updated_node.func, cst.Name):
            if updated_node.func.value == "cast":
                # cast(Type, value) -> value
                if len(updated_node.args) >= 2:
                    return updated_node.args[1].value
            elif updated_node.func.value == "assert_never":
                # assert_never(x) -> assert False
                return cst.Assert(test=cst.Name("False"))
        
        return updated_node

    def leave_Subscript(self, original_node, updated_node):
        """Handle generic type subscripts."""
        # Skip subscripts that are array/list slicing (not types)
        # These usually have Slice or integer indices, not just names
        if any(isinstance(elem.slice, cst.Slice) for elem in updated_node.slice):
            return updated_node
            
        # Special handling for ClassVar - keep it but replace its type parameter with Any
        if isinstance(updated_node.value, cst.Name) and updated_node.value.value == "ClassVar":
            self.needs_any_import = True
            # Replace ClassVar[T] with ClassVar[Any]
            return updated_node.with_changes(
                slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Name("Any")))]
            )
        
        # Special handling for known type constructors
        if isinstance(updated_node.value, cst.Name):
            if updated_node.value.value in (
                "List", "Dict", "Set", "Tuple", "Optional", "Union",
                "Callable", "Type", "Iterable", "Iterator", "Sequence",
                "Mapping", "Generic", "TypeVar", "Literal", "Annotated",
                "TypeAlias", "ParamSpec", "TypeVarTuple", "Unpack"
            ):
                # These are typing constructs - replace with Any
                self.needs_any_import = True
                return cst.Name("Any")
        
        # For other subscripts (like Var[T] or general array access), we'll let the context handle them
        return updated_node

    def leave_Lambda(self, original_node, updated_node):
        """Handle lambda functions with type annotations."""
        # Remove parameter annotations from lambda
        new_params = updated_node.params.with_changes(
            params=[p.with_changes(annotation=None) for p in updated_node.params.params],
        )
        return updated_node.with_changes(params=new_params)


def strip_types_from_file(input_path: Path, output_path: Path) -> None:
    """Strip types from a single Python file."""
    try:
        source_code = input_path.read_text()
        
        # Skip if file is empty or contains only whitespace
        if not source_code.strip():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(source_code)
            return
        
        tree = cst.parse_module(source_code)
        transformer = TypeStripper()
        modified_tree = tree.visit(transformer)
        
        # If we need to import Any and it's not already imported, add the import
        if transformer.needs_any_import and not transformer.has_any_import:
            # Find the position to insert the import
            insert_position = 0
            for i, stmt in enumerate(modified_tree.body):
                # Insert after future imports
                if isinstance(stmt, cst.SimpleStatementLine):
                    for expr in stmt.body:
                        if isinstance(expr, cst.ImportFrom):
                            if expr.module and expr.module.value == "__future__":
                                insert_position = i + 1
                # Stop at first non-import statement
                elif not isinstance(stmt, (cst.SimpleStatementLine, cst.EmptyLine)):
                    break
            
            # Create the import statement
            import_any = cst.SimpleStatementLine(
                body=[cst.ImportFrom(
                    module=cst.Name("typing"),
                    names=[cst.ImportAlias(name=cst.Name("Any"))]
                )]
            )
            
            # Insert the import
            new_body = list(modified_tree.body)
            new_body.insert(insert_position, import_any)
            modified_tree = modified_tree.with_changes(body=new_body)
        
        stripped_code = modified_tree.code
        
        # Write the stripped code
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(stripped_code)
        print(f"Processed: {input_path} -> {output_path}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        # Copy the file as-is if we can't process it
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)


def strip_types_from_directory(input_dir: Path, output_dir: Path) -> None:
    """Strip types from all Python files in a directory."""
    python_files = list(input_dir.glob("**/*.py"))
    
    print(f"Found {len(python_files)} Python files to process")
    
    for py_file in python_files:
        relative_path = py_file.relative_to(input_dir)
        output_path = output_dir / relative_path
        strip_types_from_file(py_file, output_path)
    
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
    """Strip type annotations from Python source files.
    
    Args:
        input_dir: Directory containing Python files to process.
        output_dir: Directory where stripped files will be written.
    """
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        return
    
    print(f"Stripping types from {input_dir} to {output_dir}")
    strip_types_from_directory(input_dir, output_dir)
    print("Done!")


if __name__ == "__main__":
    tyro.cli(main)
