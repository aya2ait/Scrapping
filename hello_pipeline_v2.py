from kfp import dsl, compiler

# Define a simple component
@dsl.component
def say_hello(name: str) -> str:
    print(f"Hello {name}!")
    return f"Hello {name}!"

# Define the pipeline
@dsl.pipeline(
    name='Hello World Pipeline V2',
    description='A simple test pipeline compatible with KFP v2.0.0'
)
def hello_pipeline_v2(name: str = "World"):
    # Use the component
    say_hello_task = say_hello(name=name)

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=hello_pipeline_v2,
        package_path='hello_pipeline_v2.yaml'
    )

