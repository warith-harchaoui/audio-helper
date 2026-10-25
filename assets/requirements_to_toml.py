"""
requirements_to_toml — one-shot helper that converts a legacy
``requirements.txt`` into the ``[tool.poetry.dependencies]`` block of a
``pyproject.toml``. Kept in ``assets/`` because it is a build-time
utility, not part of the ``audio_helper`` public API.

Usage Example
-------------
>>> #   python assets/requirements_to_toml.py \\
>>> #       --requirements_file requirements.txt \\
>>> #       --project_name audio-helper \\
>>> #       --description "Audio Helper" \\
>>> #       --authors "Warith HARCHAOUI <warithmetics@deraison.ai>" \\
>>> #       --python_version "^3.10" --opensource

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

import os

import click

def read_requirements(file_path='requirements.txt'):
    with open(file_path, 'r') as f:
        requirements = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
    return requirements

def generate_pyproject_toml(requirements, project_name, description, authors, python_version, opensource:bool=False):    

    if opensource:
        license = 'BSD-2-Clause'
    else:
        license = ''

    return f"""
[tool.poetry]
name = "{project_name}"
version = "1.0.0"
description = "{description}"
authors = {authors}
license = "{license}"
readme = "README.md"

[tool.poetry.dependencies]
python = "{python_version}"
{format_dependencies(requirements)}

[tool.poetry.dev-dependencies]
pytest = "^6.2.5"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
"""

def format_dependencies(requirements):
    dependencies = ""
    for requirement in requirements:
        # Handle specific version constraints
        if '==' in requirement:
            package, version = requirement.split('==')
            dependencies += f'    {package.strip()} = "{version.strip()}"\n'
        elif '>=' in requirement:
            package, version = requirement.split('>=')
            dependencies += f'    {package.strip()} = ">={version.strip()}"\n'
        elif '<=' in requirement:
            package, version = requirement.split('<=')
            dependencies += f'    {package.strip()} = "<={version.strip()}"\n'
        elif '@ git+' in requirement:  # Handling Git URL dependencies            
            package, u = requirement.split('@ git+')
            u = u.replace("git@github.com:", "https://github.com/")
            u = u.replace('https://https://', 'https://').strip()
            if '@' in u:
                u = u.split('@')[0]
            dependencies += f'    {package.strip()} = {{ git = "{u}", branch = "main" }}\n'
        else:  # Default to wildcard for unversioned or unspecified dependencies
            dependencies += f'    {requirement.strip()} = "*"\n'
    return dependencies

def write_pyproject_toml(content, file_path='pyproject.toml'):
    with open(file_path, 'w') as f:
        f.write(content)

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option('--project_name', default='my_project', show_default=True, help="The name of your project")
@click.option('--description', default='A sample Python project', show_default=True, help="A description of your project")
@click.option('--authors', default='Your Name <you@example.com>', show_default=True, help="List of authors")
@click.option('--python_version', default='^3.8', show_default=True, help="Python version to be used in the project")
@click.option('--requirements_file', default='requirements.txt', show_default=True, help="Path to requirements.txt file")
@click.option('--output_file', default='pyproject.toml', show_default=True, help="Path to output pyproject.toml file")
@click.option("--opensource", is_flag=True, help="Use this flag if your project is open source")
def main(project_name, description, authors, python_version, requirements_file, output_file, opensource):
    """Generate pyproject.toml from requirements.txt."""
    authors = [author.strip() for author in authors.split(',')]
    authors = [a for a in authors if len(a) > 0]

    if not python_version.startswith('^'):
        python_version = f'^{python_version}'

    if not os.path.exists(requirements_file):
        print(f"Error: {requirements_file} not found.")
    else:
        requirements = read_requirements(requirements_file)
        toml_content = generate_pyproject_toml(
            requirements,
            project_name=project_name,
            description=description,
            authors=authors,
            python_version=python_version,
            opensource=opensource
        )
        write_pyproject_toml(toml_content, file_path=output_file)
        print(f"pyproject.toml has been generated successfully at {output_file}.")

if __name__ == '__main__':
    main()
