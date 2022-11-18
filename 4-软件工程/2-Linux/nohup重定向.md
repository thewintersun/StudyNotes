### Redirecting Output to a File

By default, `nohup` redirects the command output to the `nohup.out` file. If you want to redirect the output to a different file, use the standard shell redirection.

For example, to redirect the standard output and standard error to the `mycommand.out` you would use:

```
nohup mycommand > mycommand.out 2>&1 &
```

To redirect the standard output and standard error to different files:

```
nohup mycommand > mycommand.out 2> mycommand.err &
```

