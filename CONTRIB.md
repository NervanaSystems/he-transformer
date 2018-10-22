Contributor Guidelines
======================

License
-------

All contributed code must be compatible with the [Apache
2](https://www.apache.org/licenses/LICENSE-2.0) license, preferably by
being contributed under the Apache 2 license. Code contributed with
another license will need the license reviewed by Intel before it can be
accepted.

Code formatting
---------------

All C/C++ source code in the repository, including the test code, must
adhere to the source-code formatting and style guidelines described
here. The coding style described here applies to the nGraph-he repository.
Related repositories may make adjustements to better match the coding
styles of libraries they are using.


#### Formatting

Things that look different should look different because they are
different. We use **clang format** to enforce certain formatting.
Although not always ideal, it is automatically enforced and reduces
merge conflicts.

-   The .clang-format file located in the root of the project specifies
    our format.
    -   The script maint/apply-code-format.sh enforces that formatting
        at the C/C++ syntactic level.
    -   The script at maint/check-code-format.sh verifies that the
        formatting rules are met by all C/C++ code (again, at the
        syntax level). The script has an exit code of `0` when code
        meets the standard and non-zero otherwise. This script does
        *not* modify the source code.

