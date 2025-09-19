from agenesis.perception import TextPerception, PerceptionResult, InputType


def test_text_perception_basic():
    perception = TextPerception()
    result = perception.process("Hello world")
    
    assert isinstance(result, PerceptionResult)
    assert result.content == "Hello world"
    assert result.input_type == InputType.TEXT


def test_text_perception_invalid_input():
    perception = TextPerception()
    
    assert not perception.validate_input(123)
    assert not perception.validate_input("")