use ::std::{
    boxed::Box,
    clone::Clone,
    iter::Iterator,
    marker::Send,
    ops::{Drop, FnOnce},
    sync::{
        mpsc::{channel, Receiver, Sender},
        {Arc, Mutex},
    },
    thread::{self, JoinHandle},
    vec::Vec,
};

pub struct Task {
    f: Box<dyn FnOnce() + Send + 'static>,
}

impl Task {
    pub fn new<F: FnOnce() + Send + 'static>(f: F) -> Self {
        Task { f: Box::new(f) }
    }
}

pub trait Executor {
    fn execute(&mut self, task: Task);
}

pub struct ThreadPool {
    threads: Vec<JoinHandle<()>>,
    sender: Sender<Task>,
}

impl ThreadPool {
    pub fn new(size: usize) -> Self {
        let (sender, receiver): (Sender<Task>, Receiver<Task>) = channel();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut threads = Vec::with_capacity(size);
        for _ in 0..size {
            let receiver = Arc::clone(&receiver);
            threads.push(thread::spawn(move || loop {
                let task = receiver.lock().unwrap().recv().unwrap();
                (task.f)();
            }));
        }
        ThreadPool { threads, sender }
    }
}

impl Executor for ThreadPool {
    fn execute(&mut self, task: Task) {
        self.sender.send(task).unwrap()
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.threads.drain(..).for_each(|t| t.join().unwrap());
    }
}
