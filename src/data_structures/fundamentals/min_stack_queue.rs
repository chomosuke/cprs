/*
 * min stack queue
 * you have a pointer pointing towards the smallest element in the queue.
 * then, each smallest element mus point towards the next smallest element.
 *
 * Now the problem arise whe there's 3 smallest element, each is the next smallest element of the
 * privous one.
 * I guess, whenever you dequeue the smallest element and don't know the next smallest, you just
 * scan the queue again, but, this doesn't have even constant time amortized cost...
 *
 * Here's what you do, you have a stack.
 *
 * Okay, instead of a stack, you have a doubly linked list of elements.
 *
 * The doubly linked list start with the minimum element of the queue.
 *
 * Then, the minimum elemen of the queue will point towards the smallest element of the queue
 * behind it.
 *
 * so, the last element of the queue is definitely a member of this doubly linked list, because,
 * it's at least the smallest element of itself.
 *
 * Anyways, when you insert an element, if you're smaller than the previous elemest of the queue,
 * you should find the previous previous element of the list, and keep going until you find an
 * element that's smaller than you, and break the linked list so that the closest previous element
 * smaller than you now point towards you.
 *
 * When you pop the queue, you already have the next smallest element.
 * So, obivously this work, but how does this amortize?
 *
 * Well, note that every element will enter the list.
 * but they only leave the list once, they never re-enter the list.
 *
 * Hence, this is amortized to be constant.
 *
 * I think there might be a simpler solution I'm missig?
 *
 * Hmmmmm, maybe I can prove it's not possible?
 *
 * Look, everytime you pop an element from the queue, you must already know the next smallest
 * element.
 *
 * yeah, this kinda feels like the best way to solve this, I have a hunch that if I try I can
 * actually prove this but that will take a lot of effort so I won't do that.
 *
 * heck yeah
 *
 * I was right
 *
 * okay, I guess my idea actually is just the same as maintaining two queue.
 *
 *
 *
 * */

mod min_stack {
    use std::{cmp::Ord, rc::Rc};

    struct Elem<T> {
        value: Rc<T>,
        min_below: Rc<T>,
    }
    pub struct MinStack<T>(Vec<Elem<T>>);

    impl<T: Ord> MinStack<T> {
        pub fn new() -> Self {
            Self(Vec::new())
        }

        pub fn push(&mut self, value: T) {
            let vec: &mut Vec<_> = &mut self.0;
            let value = Rc::new(value);

            let min_below: Option<Rc<T>> = vec
                .last()
                .map(|Elem { min_below, .. }| Rc::clone(min_below));

            let min_below = if let Some(min_below) = min_below {
                min_below.min(Rc::clone(&value))
            } else {
                Rc::clone(&value)
            };

            vec.push(Elem { value, min_below });
        }

        pub fn pop(&mut self) -> Option<T> {
            let vec: &mut Vec<_> = &mut self.0;
            vec.pop()
                .map(|Elem { value, .. }| Rc::into_inner(value).unwrap())
        }

        pub fn min(&self) -> Option<&T> {
            let vec: &Vec<_> = &self.0;
            vec.last().map(|Elem { min_below, .. }| &**min_below)
        }

        pub fn is_empty(&self) -> bool {
            self.0.is_empty()
        }
    }
}

mod min_queue_stacked {
    use super::min_stack::*;
    use std::cmp::Ord;

    pub struct MinQueue<T> {
        stack1: MinStack<T>,
        stack2: MinStack<T>,
    }
    impl<T: Ord> MinQueue<T> {
        pub fn new() -> Self {
            Self {
                stack1: MinStack::new(),
                stack2: MinStack::new(),
            }
        }

        pub fn push(&mut self, value: T) {
            self.stack1.push(value);
        }

        pub fn pop(&mut self) -> Option<T> {
            let Self { stack1, stack2 } = self;

            if stack2.is_empty() {
                while let Some(v) = stack1.pop() {
                    stack2.push(v);
                }
            }

            stack2.pop()
        }

        pub fn min(&self) -> Option<&T> {
            let Self { stack1, stack2 } = self;

            match (stack1.min(), stack2.min()) {
                (Some(min1), Some(min2)) => Some(min1.min(min2)),
                (Some(min), None) | (None, Some(min)) => Some(min),
                (None, None) => None,
            }
        }
    }
}

mod min_queue_linked {}
