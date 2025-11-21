from fastapi import FastAPI


api = FastAPI()


@api.get("/add")
def add_numbers(a: int, b: int):
    return {"result": {a+b}}
