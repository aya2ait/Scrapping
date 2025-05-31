"""
ASCII-only Kubeflow Pipeline
"""

from kfp import dsl, compiler
from kfp.dsl import component, pipeline

@component(base_image="python:3.9-slim")
def hello() -> str:
    """Simple hello world component"""
    print("Hello World")
    return "success"

@pipeline(name="ascii-test-pipeline")
def ascii_pipeline():
    """ASCII-only pipeline with no parameters"""
    hello()

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=ascii_pipeline,
        package_path="ascii_pipeline.yaml"
    )
    print("Pipeline compiled to ascii_pipeline.yaml")