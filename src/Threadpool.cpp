#include <thread>
#include <mutex>
#include <iostream>
#include <future>

#include "Threadpool.hpp"

ThreadPool::ThreadPool(std::size_t num_threads)
    : activeThreads(0)
{
    Start(num_threads);
};

ThreadPool::~ThreadPool()
{
    Stop();
}

void ThreadPool::Start(std::size_t num_threads)
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        if (threads.size())
            throw std::runtime_error("attempted to reinitialize threadpool during runtime");

        const std::size_t max_threads = std::thread::hardware_concurrency();

        if (num_threads > max_threads) {
            std::cout << num_threads << " exceeds max number of available threads, booting " << max_threads << " threads instead" << std::endl;
            num_threads = max_threads;
        }
        
        for (unsigned int i = 0; i < num_threads; ++i) {
            threads.emplace_back(std::thread(&ThreadPool::ThreadLoop, this));
        }

        should_terminate = false;
    }
}

void ThreadPool::ThreadLoop()
{
    while (true) {
        Task task;
        
        // Brackets are used to manage lock scope, disposed once task is extracted from queue
        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            mutex_condition.wait(lock, [this] {
                return !tasks.empty() || should_terminate;
            });

            if (should_terminate)
                return;

            task = tasks.front();
            activeThreads.fetch_add(1);
            tasks.pop();
        }
        
        try
        {
            task();
            activeThreads.fetch_sub(1);
        }
        catch(const std::exception& error)
        {
            std::cerr << error.what() << std::endl;
            throw std::runtime_error("threadpool task error");
        }
    }
}

void ThreadPool::QueueTask(const Task &task)
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.push(task);
    }

    mutex_condition.notify_one();
}

std::size_t ThreadPool::poolSize()
{
    std::size_t size;

    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        size = threads.size();
    }

    return size;
}

bool ThreadPool::busy() {
    bool poolbusy;

    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        poolbusy = !tasks.empty() || activeThreads.load() != 0;
    }

    return poolbusy;
}

void ThreadPool::WaitToFinish()
{
    // Wait for all currently executing tasks to finish
    while (busy()) {
        std::this_thread::yield(); // Give up the CPU to avoid busy-waiting
    }
}

void ThreadPool::Stop() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        should_terminate = true;
    }

    mutex_condition.notify_all();

    for (std::thread& active_thread : threads) {
        active_thread.join();
    }

    threads.clear();
}