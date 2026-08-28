# AGENTS.md

## 1. Project Overview

This repository contains numerical simulation and analysis code for trapped-ion laser spectroscopy.

The main purposes of the project are:

- simulate trapped-ion dynamics,
- calculate ion positions and velocities as functions of time,
- evaluate laser-ion interaction,
- calculate excitation probabilities and spectroscopy signals,
- perform FFT analysis of ion motion,
- compare numerical results with theoretical normal-mode frequencies,
- save simulation results and corresponding physical parameters for later analysis.

The code is primarily written in Python and is intended to run on Windows.

---

## 2. General Editing Policy

When modifying this repository:

1. Preserve the existing physical model unless the user explicitly requests a change.
2. Prefer minimal modifications over large-scale refactoring.
3. Do not rename physical variables unless explicitly requested.
4. Do not change numerical parameters, physical constants, or initial conditions unless explicitly requested.
5. Do not remove existing output data, plots, or saved parameters unless explicitly requested.
6. Preserve compatibility with the current execution environment whenever possible.
7. Before making a large or multi-file modification, identify:
   - the files that need modification,
   - the reason each file must be modified,
   - the expected effect of the modification.
8. Avoid unrelated cleanup or stylistic changes when fixing a specific problem.

When the user requests a "minimal change", modify only the code necessary to satisfy the request.

---

## 3. Source of Truth

The files currently present in this repository are the source of truth.

Do not rely on old code snippets, previous discussions, generated examples, or assumptions when they conflict with the current repository.

Before changing behavior, inspect the current implementation and its call sites.

If documentation and source code conflict, treat the source code as authoritative unless the user explicitly states otherwise.

---

## 4. Physical Model Protection

The following parts of the simulation should be treated as part of the physical model.

Do not modify them implicitly.

Examples include:

- ion mass,
- trap frequency,
- Coulomb interaction,
- harmonic trapping force,
- radiation-pressure cooling force,
- recoil heating,
- laser wavelength,
- wave number,
- linewidth,
- saturation parameter,
- detuning definitions,
- optical Bloch equation implementation,
- excitation probability,
- Doppler shift calculation,
- initial position and velocity conditions,
- random processes and random seeds,
- time step definitions,
- normal-mode calculations.

If a requested software optimization could change the physical or numerical result, explicitly state the possible effect before applying the modification.

Performance optimization must not silently change the mathematical model.

---

## 5. Numerical Correctness

Preserve numerical behavior whenever possible.

When changing numerical code:

- preserve units,
- preserve array shapes unless explicitly changing the data structure,
- preserve indexing conventions,
- preserve time-step definitions,
- preserve coordinate conventions,
- preserve frequency definitions,
- preserve normalization conventions.

Pay special attention to:

- Hz versus rad/s,
- MHz versus Hz,
- nm versus m,
- angular frequency omega versus ordinary frequency f,
- array axis order,
- particle index versus time index,
- detuning sign conventions.

Do not assume two quantities with similar names use the same units.

When changing numerical algorithms, explain whether the result should be:

- mathematically identical,
- numerically equivalent within floating-point error,
- approximately equivalent,
- or physically modified.

---

## 6. Parameter Policy

Parameters should be treated according to their role.

Possible categories include:

- physical constants,
- experiment-dependent parameters,
- simulation-control parameters,
- analysis-only parameters.

Rules:

- Never change physical constants automatically.
- Change experiment-dependent parameters only when explicitly requested.
- Simulation-control parameters may be reduced temporarily only when explicit execution or testing has been authorized.
- Do not permanently overwrite production parameter values merely to make a test easier to run.
- Do not silently change analysis-only parameters if doing so changes interpretation of plots, spectra, or statistics.

If an error appears to be caused by an invalid or impractical parameter value:

1. identify the parameter,
2. explain the limitation,
3. propose a reasonable alternative or test value,
4. leave the production value unchanged unless explicitly instructed otherwise.

---

## 7. Memory Usage

This project may handle very large NumPy arrays.

Arrays such as position, velocity, density-matrix components, spectra, and parameter sweeps may require several GB of RAM.

Therefore:

1. Avoid unnecessary copies of large NumPy arrays.
2. Avoid converting large memory-mapped arrays into ordinary in-memory arrays unless necessary.
3. Prefer views, slicing, generators, chunking, or memory mapping when appropriate.
4. Do not use `.copy()` on large arrays without a clear reason.
5. Consider dtype size when allocating large arrays.
6. Estimate memory requirements before introducing a large allocation.
7. Avoid sending very large arrays unnecessarily between multiprocessing workers.
8. Consider temporary-array memory consumption in addition to the final array size.

When a change significantly affects memory usage, estimate the approximate additional or reduced memory consumption.

---

## 8. Multiprocessing and Windows

The project may use Python multiprocessing on Windows.

When editing multiprocessing code:

- preserve compatibility with Windows multiprocessing,
- ensure process creation is protected by:

```python
if __name__ == "__main__":
    ...
```

- avoid relying on Unix-only `fork` behavior,
- ensure worker functions can be serialized when required,
- minimize copying of large arrays between processes,
- avoid creating excessive worker processes,
- preserve the user-configurable `n_workers` mechanism when present.

Do not increase the default number of workers solely for performance unless explicitly requested.

When multiprocessing and memory usage conflict, prioritize system stability over maximum CPU utilization.

---

## 9. Simulation Execution Safety

Some full simulations can consume large amounts of RAM, CPU time, and disk space.

Do not automatically launch a full-scale simulation after modifying code.

Do not automatically launch even a reduced simulation unless execution has been explicitly authorized.

If execution is explicitly requested, prefer a reduced test configuration when appropriate, such as:

- fewer detuning points,
- fewer ions,
- shorter simulation time,
- fewer recorded time steps,
- smaller parameter sweeps.

Do not modify the user's production parameter values permanently for testing.

If temporary test parameters are used, clearly distinguish them from production parameters.

Before running a potentially expensive calculation, inspect the expected workload whenever practical.

---

## 10. File and Data Safety

Simulation output may contain expensive-to-reproduce numerical results.

Therefore:

- do not delete existing output files,
- do not overwrite existing simulation results unless explicitly requested,
- preserve existing directory structures unless necessary,
- prefer creating a new output file when testing,
- do not rename saved-data fields without checking compatibility with analysis scripts.

When changing the save format, check compatibility with existing loading and plotting programs.

Existing scientific output should be treated as read-only unless explicitly stated otherwise.

---

## 11. Parameter Metadata

Simulation results should remain associated with the physical parameters used to produce them.

When editing save/load code, preserve metadata such as:

- trap frequency,
- laser-related parameters,
- simulation time step,
- recording time step,
- random seed,
- particle number,
- physical constants or configurable physical parameters used by the simulation.

Do not silently introduce a parameter that affects results without considering whether it should also be saved as metadata.

JSON may be used for human-readable simulation metadata when that is the existing project convention.

---

## 12. Plotting and FFT Analysis

When modifying plotting or FFT-related code:

- preserve physical axis definitions,
- preserve units,
- preserve existing particle-by-particle plotting behavior unless explicitly requested,
- preserve frequency ranges unless explicitly changed,
- do not change normalization silently,
- do not change FFT conventions without explanation.

When comparing FFT peaks to theoretical normal modes, keep theoretical and numerical quantities in consistent frequency units.

Avoid changing plot appearance unless the user requested presentation changes.

---

## 13. Code Review Procedure

When asked to review code, use the following procedure.

### Step 1: Identify the problem

State:

- affected file,
- affected function or code section,
- likely cause.

### Step 2: Evaluate impact

Classify the issue when relevant as:

- correctness,
- physical-model error,
- numerical error,
- memory issue,
- performance issue,
- multiprocessing issue,
- file I/O issue,
- plotting issue,
- maintainability issue.

### Step 3: Propose the smallest solution

Prefer the least invasive modification that solves the problem.

### Step 4: Explain consequences

State whether the modification changes:

- numerical results,
- physical results,
- memory usage,
- execution time,
- saved-data format,
- backward compatibility.

### Step 5: Modify

Only after identifying the above, apply the necessary changes.

---

## 14. Debugging Procedure

When an error occurs:

1. Read the complete traceback.
2. Identify the first relevant error in project code.
3. Trace the data or function flow leading to the error.
4. Determine whether the cause is:
   - syntax,
   - type,
   - shape,
   - indexing,
   - memory,
   - disk,
   - multiprocessing,
   - dependency,
   - parameter configuration,
   - physical/numerical implementation.
5. Fix the root cause rather than suppressing the exception.

Do not add broad `try/except` blocks simply to hide errors.

---

## 15. Refactoring Policy

Do not perform large refactoring unless explicitly requested.

In particular, do not automatically:

- split files,
- merge files,
- convert procedural code to classes,
- rename major functions,
- restructure the project,
- introduce new frameworks,
- replace NumPy with another numerical library,
- change multiprocessing architecture.

A bug fix or performance improvement should normally preserve the current project structure.

If substantial refactoring would provide a meaningful benefit, propose it separately from the immediate fix.

---

## 16. Dependency Policy

Avoid introducing new Python packages when the task can reasonably be solved using existing dependencies.

Common project dependencies may include:

- Python,
- NumPy,
- Matplotlib,
- standard-library multiprocessing,
- standard-library JSON and file I/O.

Before adding another dependency:

1. explain why it is needed,
2. explain why existing dependencies are insufficient,
3. consider Windows compatibility.

Do not install or upgrade dependencies automatically.

---

## 17. Communication Format

For code modifications, report changes in the following format whenever practical:

### Changed files

- `filename.py`
  - changed function or section
  - reason for change

### Behavioral effect

Explain what changes from the user's perspective.

### Physical/numerical effect

State explicitly whether the physical model or numerical result changes.

### Performance effect

State expected effects on:

- RAM,
- CPU usage,
- execution time,
- disk usage.

### Validation

Describe how the change was checked.

---

## 18. Priority Order

When requirements conflict, use the following priority order:

1. Correct physical model
2. Correct numerical result
3. Preservation of existing data
4. System stability
5. Backward compatibility
6. Minimal code modification
7. Performance
8. Code style and cosmetic cleanup

Performance improvements must not take priority over physical or numerical correctness.

---

## 19. Project-Specific Entry Points

The main simulation entry point is expected to be:

- `main.py`

Analysis, plotting, FFT, simulation, and save/load functionality may exist in separate modules.

Before modifying a function, inspect where it is called from and how its return values are used.

Do not assume a file is independent simply because it can be executed individually.

If the actual project structure differs from this section, update this section to match the repository.

---

## 20. Change History and Validation Policy

The AI agent may inspect and modify project files, but it must not create permanent repository history or publish changes unless explicitly instructed.

### Git operations

Do not perform any of the following unless explicitly requested:

- `git commit`
- `git push`
- `git pull`
- `git merge`
- `git rebase`
- `git reset`
- `git checkout` or `git switch` when it changes the current working state
- branch creation or deletion
- tag creation or deletion
- modification of remote repository settings

The preferred workflow is:

1. inspect the existing code,
2. make the requested file changes,
3. show or summarize the resulting diff,
4. validate the modification,
5. leave the final commit decision to the user.

Do not automatically commit changes after successful validation.

When reporting completed work, clearly distinguish:

- files modified,
- files created,
- files deleted,
- validation performed,
- validation not performed.

Never describe a change as validated unless an actual validation step was performed.

---

## 21. FILE-EDIT-ONLY Default Mode

Unless explicitly instructed otherwise, operate in **FILE-EDIT-ONLY mode**.

FILE-EDIT-ONLY mode permits:

- reading files inside the current project,
- searching within project files,
- inspecting source code,
- inspecting configuration files,
- inspecting Git status or diffs without modifying repository state,
- creating or modifying source-code and documentation files required by the user's request.

FILE-EDIT-ONLY mode does **not** permit:

- running project code,
- running simulations,
- running tests,
- launching terminal commands with side effects,
- changing Git history,
- installing or removing packages,
- modifying the Python environment,
- modifying IDE or operating-system settings,
- deleting, moving, or renaming files unless explicitly requested,
- overwriting scientific output data,
- accessing external networks,
- uploading or downloading project data,
- starting background processes.

File modification permission does not imply execution permission.

---

## 22. Command Execution Policy

Do not execute programs, scripts, shell commands, or terminal commands unless the user explicitly requests execution.

This includes:

- Python scripts,
- simulation programs,
- analysis scripts,
- test suites,
- build commands,
- package-manager commands,
- Git commands that modify repository state,
- system utilities.

Read-only inspection commands may be used only when necessary to understand the project and when they do not modify system or repository state.

Examples of generally acceptable read-only inspection include:

```text
git status
git diff
git log
```

Do not assume that a command is safe merely because it is commonly used during development.

If there is uncertainty about whether a command has side effects, do not execute it.

---

## 23. Simulation and Analysis Execution

Do not automatically execute:

- full simulations,
- reduced simulations,
- parameter sweeps,
- FFT analyses,
- plotting scripts,
- data conversion scripts,
- benchmark programs,
- test calculations.

Code modification and code execution are separate permissions.

A request to "fix", "modify", "implement", or "review" code does not imply permission to run it.

Only execute code when the user explicitly requests execution or validation by execution.

When execution is explicitly requested, prefer the minimum workload necessary to validate the requested behavior.

---

## 24. File Operation Restrictions

Unless explicitly requested, do not:

- delete files,
- move files,
- rename files,
- overwrite existing output data,
- modify files outside the current project,
- create files outside the current project,
- change directory structures,
- modify generated scientific data,
- modify archived simulation results,
- alter binary files,
- alter external datasets.

Existing simulation output must be treated as read-only scientific data unless the user explicitly requests otherwise.

When a new file is necessary to implement the requested code change, explain why it is required.

Prefer modifying an existing appropriate file over creating unnecessary new files.

---

## 25. Environment Modification Policy

Do not modify the development or operating-system environment unless explicitly requested.

This includes:

- installing Python packages,
- uninstalling packages,
- upgrading packages,
- downgrading packages,
- modifying Python environments,
- creating or deleting virtual environments,
- modifying PATH,
- modifying environment variables,
- changing IntelliJ or IDE settings,
- changing interpreter settings,
- changing operating-system configuration.

Do not run:

```text
pip install
pip uninstall
conda install
conda update
conda remove
pacman
winget
choco
```

or equivalent package-management commands unless explicitly requested.

If a missing dependency is identified, report the dependency and the recommended installation command, but do not execute the installation automatically.

---

## 26. External Communication and Network Policy

Do not perform external network actions unless explicitly requested.

This includes:

- uploading files,
- downloading files,
- sending HTTP requests,
- accessing external APIs,
- sending email or messages,
- publishing results,
- pushing code to remote repositories,
- accessing cloud storage,
- transferring simulation data.

Source code and research data must not be transmitted to an external service solely for convenience.

If an operation requires external network access, identify that requirement before performing it.

---

## 27. Process and Resource Policy

Do not start persistent or background processes unless explicitly requested.

This includes:

- servers,
- GUI applications,
- watchers,
- continuous monitoring processes,
- long-running simulations,
- background Python processes.

Do not terminate existing user processes unless explicitly requested.

Do not change:

- process priorities,
- CPU affinity,
- multiprocessing worker limits,
- memory limits,

unless explicitly requested.

System stability takes priority over maximizing computational performance.

---

## 28. Scientific Data Protection

Treat generated simulation data as research assets.

Unless explicitly requested, do not:

- overwrite data,
- truncate data,
- convert data destructively,
- normalize stored data in place,
- remove metadata,
- modify timestamps or parameter records,
- replace original files with processed versions.

For transformations or analysis, prefer creating derived outputs while preserving the original data.

Any operation capable of destroying or replacing scientific data requires explicit user instruction.

---

## 29. Configuration Protection

Do not modify project-wide configuration unless explicitly requested.

Protected configuration includes, but is not limited to:

- Python interpreter configuration,
- IntelliJ project configuration,
- run/debug configurations,
- Git configuration,
- `.gitignore`,
- environment-variable files,
- dependency files,
- build configuration,
- CI/CD configuration.

If a requested source-code change requires a configuration change, explain the required change separately before applying it.

---

## 30. Validation Levels

Use the following validation terminology.

### Level 0 — Inspection only

The code was reviewed statically. No execution was performed.

### Level 1 — Structural validation

The modification was checked for:

- syntax consistency,
- imports,
- variable names,
- function signatures,
- array-shape consistency where inferable.

No program execution was performed.

### Level 2 — Limited execution validation

A small or reduced test was explicitly authorized and executed.

Report the exact test configuration used.

### Level 3 — Production-scale validation

The actual production calculation was executed.

This level requires explicit user instruction.

Never imply Level 2 or Level 3 validation when only static inspection was performed.

---

## 31. Explicit Authorization Rule

Permission should be interpreted narrowly.

For example:

`Modify this function.`

authorizes file modification but does not authorize:

- running the program,
- installing dependencies,
- committing the change,
- pushing to GitHub,
- deleting old files,
- changing configuration.

`Test this modification.`

authorizes appropriate test execution but does not automatically authorize:

- full production simulation,
- package installation,
- external network access,
- Git commits.

When an action is not clearly authorized, prefer a non-destructive, local, file-only approach.

---

## 32. Scientific Transparency

For any modification that may affect scientific interpretation, clearly distinguish among:

- software implementation changes,
- numerical-method changes,
- physical-model changes,
- parameter changes.

Never describe a numerical approximation as physically exact.

When uncertain about the intended physical interpretation, preserve the existing implementation and flag the ambiguity rather than silently choosing a new interpretation.
