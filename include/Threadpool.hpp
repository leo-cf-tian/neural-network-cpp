#pragma once

#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <atomic>

/**
 * Threadpool implementation from https://stackoverflow.com/questions/15752659/thread-pooling-in-c11
 * 
 * Currently only supports tasks with no return values
*/
class ThreadPool {
    using Task = std::function<void()>;

    public:
        
        /**
         * Initializes threads with a certain number of threads, max by default
        */
        ThreadPool(std::size_t num_threads = std::thread::hardware_concurrency());
        ~ThreadPool();

        void QueueTask(const Task& task);

        /**
         * Waits until tasks are finished
        */
        void WaitToFinish();
        
        std::size_t poolSize();

    private:
        void ThreadLoop();

        bool should_terminate = false;              // Tells threads to stop looking for tasks
        std::mutex queue_mutex;                     // Prevents data races to the job queue
        std::condition_variable mutex_condition;    // Allows threads to wait on new jobs or termination 

        std::vector<std::thread> threads;
        std::queue<Task> tasks;

        std::atomic<std::size_t> activeThreads;


        void Start(std::size_t num_threads);    // Starts a certain number of threads
        void Stop();                            // Blocks more tasks from being queued while waiting for existing tasks to clear
        bool busy();                            // Checks whether any tasks are currently in queue
};