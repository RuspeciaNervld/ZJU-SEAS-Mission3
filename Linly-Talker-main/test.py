from pydantic import BaseModel

class MyModel(BaseModel):
    name: str
    age: int


def test():
    model = MyModel(name='John Doe', age=25)
    assert model.name == 'John Doe'
    assert model.age == 25
    assert model.dict() == {'name': 'John Doe', 'age': 25}
    assert model.json() == '{"name": "John Doe", "age": 25}'

test()
