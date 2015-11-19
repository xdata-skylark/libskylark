# Contributing

We welcome pull requests with new features, fixes and improvements.
Please make sure to follow these two rules to get your pill requests accepted:

- Follow our coding style guide (see `doc/style.mkd`).
- First time contributors need to fill the CLA (see `CLA.txt`) and send it to
  Ken.

Try to follow a standardized commit message style (see below). Since we are
not always consistent this is not a requirement, but a good practice.


## Commit Messages

Our commit message style is loosely based on what other open source projects
use (i.e. https://github.com/angular/angular.js/blob/master/CONTRIBUTING.md#commit).
Try to follow the format:

```
<scope>: <summary>

<body>

<footer>
```

The subject line is mandatory and cannot exceed 80 characters. The `<scope>`
corresponds to the file or namesapce of the affected by the change. The
summary uses the imperative and present tense (lower case and no dot at the
end). A good rule of thumb (taken from http://chris.beams.io/posts/git-commit/)
is to think of the summary as completing the following sentence

- If applied, this commit will `<summary>`

The `<body>` should describe what and why has changed. Finally the `<footer>` is
used to reference or close issues.
