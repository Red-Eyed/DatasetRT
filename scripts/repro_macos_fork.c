/* Isolate used libdispatch semaphore inheritance without third-party libraries. */
#include <dispatch/dispatch.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
    bool warm_parent = false;
    bool fresh_child = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--warm-parent") == 0) {
            warm_parent = true;
        } else if (strcmp(argv[i], "--fresh-child") == 0) {
            fresh_child = true;
        } else {
            fprintf(stderr, "usage: %s [--warm-parent] [--fresh-child]\n", argv[0]);
            return 2;
        }
    }

    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    if (semaphore == NULL) {
        fprintf(stderr, "dispatch_semaphore_create failed\n");
        return 2;
    }
    if (warm_parent) {
        fprintf(stderr, "parent: wait before fork\n");
        dispatch_semaphore_wait(semaphore, dispatch_time(DISPATCH_TIME_NOW, 1000000));
    }

    /* No other threads are running when this probe forks. */
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 2;
    }
    if (pid == 0) {
        if (fresh_child) {
            /* Leave the inherited semaphore untouched, including its destructor. */
            semaphore = dispatch_semaphore_create(0);
            if (semaphore == NULL) {
                _exit(2);
            }
        }
        fprintf(stderr, "child: wait after fork\n");
        dispatch_semaphore_wait(semaphore, dispatch_time(DISPATCH_TIME_NOW, 1000000));
        fprintf(stderr, "child: completed\n");
        _exit(0);
    }

    int status = 0;
    pid_t waited;
    do {
        waited = waitpid(pid, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited != pid) {
        perror("waitpid");
        return 2;
    }
    dispatch_release(semaphore);
    if (WIFSIGNALED(status)) {
        fprintf(stderr, "parent: child died from signal %d\n", WTERMSIG(status));
        return 1;
    }
    if (!WIFEXITED(status)) {
        fprintf(stderr, "parent: unexpected child status %#x\n", status);
        return 2;
    }
    fprintf(stderr, "parent: child exit code %d\n", WEXITSTATUS(status));
    return WEXITSTATUS(status);
}
