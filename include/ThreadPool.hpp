#pragma once
#include <functional>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

// macOS-specific header
#ifdef __APPLE__
#include <pthread.h>
#endif


using Task = std::function<void()>;

class ThreadPool {
public:
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ThreadPool(size_t n = std::thread::hardware_concurrency())
        : stop(false), active(0), exc(nullptr)
    {
        n = std::max(size_t(1), n);
        for (size_t i = 0; i < n; i++) {
            workers.emplace_back([this]() {
                // Request high QoS for compute workers on macOS.
                // This may influence scheduling toward higher-performance cores.
                #ifdef __APPLE__
                pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
                #endif

                while(true) {
                    Task task;
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        task_cv.wait(lock, [this]() {
                            return stop || !tasks.empty();
                        });

                        if (stop && tasks.empty()) return;

                        task = std::move(tasks.front());
                        tasks.pop();
                        ++active;
                    }

                    try {
                        task();
                    } catch (...) {
                        std::unique_lock<std::mutex> lock(mtx);
                        exc = std::current_exception();
                    }

                    bool notify_allworks_done = false;
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        --active;
                        if (tasks.empty() && active == 0) {
                            notify_allworks_done = true;
                        }
                    }
                
                    if (notify_allworks_done) {
                        done_cv.notify_all();
                    }
                }
            });
        }
    }

    void submit(Task task) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            tasks.push(std::move(task));
        }
        task_cv.notify_one();
    }

    void wait_allworks_done() {
        std::unique_lock<std::mutex> lock(mtx);
        done_cv.wait(lock, [this]() { return tasks.empty() && active == 0; });
    }

    bool has_exception() const {
        std::lock_guard<std::mutex> lock(mtx);
        return exc != nullptr;
    }

    void rethrow_if_exception() {
        std::exception_ptr e;
        {
            std::lock_guard<std::mutex> lock(mtx);
            e = exc;
            exc = nullptr;
        }
        if (e) {
            std::rethrow_exception(e);
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        task_cv.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable())
                worker.join();
        }
    }

private:
    bool stop;
    int active;
    std::vector<std::thread> workers;
    std::queue<Task> tasks;
    mutable std::mutex mtx;
    std::condition_variable task_cv;
    std::condition_variable done_cv;
    std::exception_ptr exc;
};