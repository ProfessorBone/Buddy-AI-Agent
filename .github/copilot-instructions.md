# TechMentor coding guidelines (auto-injected)

**Primary context**  
• Before making any code edits, read all docstrings and class/module comments for design constraints.  
• Treat the main agent and database handler classes as the single source of truth for module boundaries and naming.  

**Testing contract**  
• For each new or modified *.py* file, create or update a test in `tests/` that drives ≥90% line coverage.  
• All tests must pass when `pytest --cov` is run.

**Commit gate**  
• Do not consider a task “done” until `flake8` and `pytest` both return exit 0.  
• If tests fail, iterate until they pass.

**Style & tooling**  
• Follow PEP8 and Black formatting for all Python code.  
• Prefer type annotations and docstrings for all public functions and classes.  
• Use descriptive variable and function names consistent with the agent's domain.
