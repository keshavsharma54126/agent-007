import typer 

app = typer.Typer()

@app.command()
def add_numbers(a:int,b:int):
    result = a+b 
    typer.echo(f"Result: {result}")
    
