import dataclasses
import json
import pytest
from tracesplit import compile_source, LanguageError
from tracesplit.language import Plan
from tracesplit.engine import run_plan, ExecutionError
from corpus import make_cases

@pytest.mark.parametrize("case", make_cases(), ids=lambda case: case["id"])
def test_constructed_corpus(case):
    if case["valid"]: assert compile_source(case["source"]).as_dict()["models"]
    else:
        with pytest.raises(LanguageError): compile_source(case["source"])

@pytest.mark.parametrize("text", ["", " ", "\n# comment\n"])
def test_empty(text):
    with pytest.raises(LanguageError): compile_source(text)

def test_unknown_symbol_has_line(source):
    with pytest.raises(LanguageError) as info: compile_source(source.replace("evaluate linear", "evaluate unknown"))
    assert info.value.line == 6

def test_immutable_plan(source):
    plan = compile_source(source)
    with pytest.raises(dataclasses.FrozenInstanceError): plan.encoded = "bad"
    p = plan.as_dict(); p["target"] = "wrong"
    assert plan.as_dict()["target"] == "target"

def test_uncompiled_plan_not_an_api_input(source, root):
    forged = Plan(compile_source(source).encoded)
    with pytest.raises(ExecutionError): run_plan(forged, root)

def test_whitespace_and_comments(source):
    text = "# leading comment\n\n"+source.replace("seed 42", "seed 42  # split seed")
    assert compile_source(text).as_dict()["split"]["seed"] == 42

@pytest.mark.parametrize("replacement", ["trees=2.5", "trees=201", "depth=31", "depth=0", "trees=1e999"])
def test_parameter_bounds(source, replacement):
    with pytest.raises(LanguageError): compile_source(source.replace("trees=40 depth=6", replacement))

@pytest.mark.parametrize("strategy", ["grouped", "temporal"])
def test_by_column_required(source, strategy):
    with pytest.raises(LanguageError): compile_source(source.replace("split stratified", "split "+strategy))
