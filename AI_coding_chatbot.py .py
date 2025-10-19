import tkinter as tk
from tkinter import scrolledtext
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

#This the Data are use in Chatbot this only gives information
data =[
    {"question": "Hello", "answer": "Hi, I am your coding AI Assistant!"},
    {"question": "GUI Interface", "answer": "• Built using Tkinter\n• Components:\n Chat Window with scroll\n Entry box for user input\n send button\n•Style:\n Right-aligned user messages,left-aligned bot reponses"},
    {"question": "Hi", "answer": "Hi, I am your coding AI Assistant!"},
    {"question": "How are you?", "answer": "I am just a program, but thanks for asking! How can I assist you today?"},
    {"question": "What can you do?", "answer": "I can help you with coding questions, generate code snippets, and provide programming tips."},
    {"question": "Can you write code for me?", "answer": "Sure! Just tell me what you need."},
    {"question": "Hello", "answer": "Hi, I am your coding AI Assistant!"},
    {"question": "How are you?", "answer": "I am just a program, but thanks for asking! How can I assist you today?"},
    {"question": "What is Python?", "answer": "Python is a high-level, interpreted programming language."},
    {"question": "What is a variable?", "answer": "A variable is a named memory location used to store data."},
    {"question": "What is AI?", "answer": "AI stands for Artificial Intelligence, which is the simulation of human intelligence in machines."},
    {"question": "What is ai?", "answer": "AI stands for Artificial Intelligence, which is the simulation of human intelligence in machines."},
    {"question": "What is ml?", "answer": "Machine learning is a subset of AI that enables systems to learn from data and improve over time without being explicitly programmed."},
    {"question": "What is machine learning?", "answer": "Machine learning is a subset of AI that enables systems to learn from data and improve over time without being explicitly programmed."},
    {"question": "What is deep learning?", "answer": "Deep learning is a type of machine learning that uses neural networks with many layers to analyze various factors of data."},
    {"question": "What is natural language processing?", "answer": "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through natural language."},
    {"question": "What is a chatbot?", "answer": "A chatbot is an AI program designed to simulate conversation with human users, especially over the Internet."},
    {"question": "What is Python?", "answer": "Python is a high-level, interpreted programming language."},
    {"question": "How to install Python?", "answer": "Download from python.org and follow the installation instructions."},
    {"question": "What is a variable in Python?", "answer": "A variable is a reserved memory location to store values."},
    {"question": "How to declare a variable in Python?", "answer": "Use assignment operator: x = 10"},
    {"question": "What is a list in Python?", "answer": "A list is a collection of items that are ordered and changeable."},
    {"question": "How to create a list in Python?", "answer": "Use square brackets: my_list = [1, 2, 3]"},
    {"question": "What is a dictionary in Python?", "answer": "A dictionary is a collection of key-value pairs."},
    {"question": "How to create a dictionary in Python?", "answer": "Use curly braces: my_dict = {'key': 'value'}"},
    {"question": "What is a loop in Python?", "answer": "A loop is used to iterate over a sequence (like a list or string)."},
    {"question": "How to write a for loop in Python?", "answer": "for i in range(5): print(i)"},
    {"question": "How to write an if statement in Python?", "answer": "if condition: do_something()"},
    {"question": "What is an exception in Python?", "answer": "An exception is an error that occurs during the execution of a program."},
    {"question": "How to handle exceptions in Python?", "answer": "Use try-except blocks: try: risky_code except Exception as e: handle_error(e)"},
    {"question": "Find the Largest number in list syntax", "answer": "max(numbers) #will return the largest number."},
    {"question": "What is the syntax to create a function in Python?", "answer": "Use def keyword: def my_function(): pass"},
    {"question": "How to create a class in Python?", "answer": "Use class keyword: class MyClass: pass"},
    {"question": "What is the difference between a list and a tuple in Python?", "answer": "Lists are mutable, tuples are immutable."},
    {"question": "How to read a file in Python?", "answer": "Use open() function: with open('file.txt', 'r') as f: content = f.read()"},
    {"question": "How to write a file in Python?", "answer": "Use open() with 'w' mode: with open('file.txt', 'w') as f: f.write('Hello World')"},
    {"question": "What is the syntax for a for loop in Python?", "answer": "for i in range(10): print(i)"},
    {"question": "What is the syntax for an if statement in Python?", "answer": "if condition: do_something()"},
    {"question": "How to import a module in Python?", "answer": "Use import keyword: import math"},
    {"question": "What is a dictionary in Python?", "answer": "It's a collection of key-value pairs."},
    {"question": "How to handle exceptions in Python?", "answer": "Use try-except blocks: try: risky_code except Exception as e: handle_error(e)"},
    {"question": "What is a lambda function in Python?", "answer": "It's an anonymous function defined with lambda keyword."},
    {"question": "How to sort a list in Python?", "answer": "Use sorted() function or list.sort() method."},
    {"question": "What is the syntax for a while loop in Python?", "answer": "while condition: do_something()"},
    {"question": "How to create a virtual environment in Python?", "answer": "Use venv module: python -m venv myenv"},
    {"question": "What is the syntax for a switch case in Python?", "answer": "Python does not have switch-case, use if-elif-else."},
    {"question": "How to concatenate strings in Python?", "answer": "Use + operator: result = str1 + str2"},
    {"question": "What is the syntax for a list comprehension in Python?", "answer": "[x for x in iterable if condition]"},
    {"question": "How to check if a key exists in a dictionary in Python?", "answer": "Use 'in' keyword: if key in my_dict: do_something()"},
    {"question": "What is the syntax for a try-except block in Python?", "answer": "try: risky_code except Exception as e: handle_error(e)"},
    {"question": "How to reverse a list in Python?", "answer": "Use list.reverse() method or slicing: reversed_list = my_list[::-1]"},
    {"question": "What is the syntax for a class method in Python?", "answer": "Use @classmethod decorator: @classmethod def my_method(cls): pass"},
    {"question": "How to create a set in Python?", "answer": "Use set() function: my_set = set([1, 2, 3])"},
    {"question": "What is the syntax for a generator function in Python?", "answer": "Use yield keyword: def my_generator(): yield value"},
    {"question": "How to check if a string contains a substring in Python?", "answer": "'substring' in my_str"},
    {"question": "What is the syntax for a map function in Python?", "answer": "Use map() function: result = map(function, iterable)"},
    {"question": "How to filter a list in Python?", "answer": "Use filter() function: result = filter(function, iterable)"},
    {"question": "What is the syntax for a reduce function in Python?", "answer": "Use functools.reduce(): from functools import reduce; result = reduce(function, iterable)"},
    {"question": "How to find the length of a list in Python?", "answer": "Use len() function: length = len(my_list)"},
    {"question": "What is the syntax for a nested loop in Python?", "answer": "for i in range(n): for j in range(m): do_something()"},
    {"question": "How to check if a number is even or odd in Python?", "answer": "Use modulus operator: if num % 2 == 0: print('Even') else: print('Odd')"},
    {"question": "What is the syntax for a list slice in Python?", "answer": "my_list[start:end:step]"},
    {"question": "How to convert a string to an integer in Python?", "answer": "Use int() function: my_int = int(my_str)"},
    {"question": "How to convert a string to a float in Python?", "answer": "Use float() function: my_float = float(my_str)"},
    {"question": "What is the syntax for a dictionary comprehension in Python?", "answer": "{key: value for item in iterable}"},
    {"question": "How to check if a list is empty in Python?", "answer": "Use if not my_list: do_something()"},
    {"question": "What is the syntax for a tuple in Python?", "answer": "Use parentheses: my_tuple = (1, 2, 3)"},
    {"question": "How to find the maximum value in a list in Python?", "answer": "Use max() function: max_value = max(my_list)"},
    {"question": "How to find the minimum value in a list in Python?", "answer": "Use min() function: min_value = min(my_list)"},
    {"question": "What is the syntax for a Python decorator?", "answer": "Use @decorator_name before function definition: @my_decorator\ndef my_function(): pass"},
    {"question": "How to check if a string starts with a substring in Python?", "answer": "Use str.startswith(): if my_str.startswith('prefix'): do_something()"},
    {"question": "How to check if a string ends with a substring in Python?", "answer": "Use str.endswith(): if my_str.endswith('suffix'): do_something()"},
    {"question": "What is the syntax for a Python context manager?", "answer": "Use with statement: with open('file.txt', 'r') as f: content = f.read()"},
    {"question": "How to create a dictionary in Python?", "answer": "Use curly braces: my_dict = {'key': 'value'}"},
    {"question": "How to iterate over a dictionary in Python?", "answer": "Use for key, value in my_dict.items(): do_something()"},
    {"question": "What is the syntax for a Python set comprehension?", "answer": "{x for x in iterable if condition}"},
    {"question": "How to reverse a string in Python?", "answer": "Use slicing: reversed = my_str[::-1]"},
    {"question": "What is a list comprehension?", "answer": "It's a compact way to create lists in Python."},
    {"question": "How to check if a string is a palindrome in Python?", "answer": "Use my_str == my_str[::-1]"},
    {"question" : "What is a Python module?", "answer": "A module is a file containing Python code that can be imported."},
    {"question": "How to import a module in Python?", "answer": "Use import keyword: import module_name"},
    {"question": "what is a library in Python?", "answer": "A library is a collection of modules that provide additional functionality."},
    {"question": "What is an API?", "answer": "An API (Application Programming Interface) allows different software applications to communicate."},
    {"question": "list the packages in Python", "answer": "Use pip list to see installed packages."},
    {"question": "What is a package in Python?", "answer": "A package is a collecation of modules oragnised in a dictory."},
    {"question" : "what is java?", "answer": "Java is a high-level, class-based, object-oriented programming language."},
    {"question": "What is a class in Java?", "answer": "A class is a blueprint for creating objects in Java."},
    {"question": "What is an object in Java?", "answer": "An object is an instance of a class."},
    {"question": "How to declare a variable in Java?", "answer": "Use data_type variable_name = value;"},
    {"question": "What is a method in Java?", "answer": "A method is a function defined inside a class."},
    {"question": "How to create a method in Java?", "answer": "Use return_type method_name(parameters) { // code }"},
    {"question": "What is inheritance in Java?", "answer": "Inheritance allows a class to inherit properties and methods from another class."},
    {"question": "What is polymorphism in Java?", "answer": "Polymorphism allows methods to do different things based on the object it is acting upon."},
    {"question": "What is encapsulation in Java?", "answer": "Encapsulation restricts direct access to an object's data and methods."},
    {"question": "What is an interface in Java?", "answer": "An interface defines a contract that classes can implement."},
    {"question": "What is an abstract class in Java?", "answer": "An abstract class cannot be instantiated and may contain abstract methods."},
    {"question": "How to handle exceptions in Java?", "answer": "Use try-catch blocks: try { // code } catch (Exception e) { // handle error }"},
    {"question": "What is a constructor in Java?", "answer": "A constructor initializes an object when it is created."},
    {"question": "How to create an array in Java?", "answer": "Use data_type[] array_name = new data_type[size];"},
    {"question": "What is a loop in Java?", "answer": "A loop allows you to execute a block of code multiple times."},
    {"question": "How to write a for loop in Java?", "answer": "for (int i = 0; i < n; i++) { // code }"},
    {"question": "How to write a while loop in Java?", "answer": "while (condition) { // code }"},
    {"question": "What is a switch statement in Java?", "answer": "A switch statement allows you to execute different parts of code based on the value of an expression."},
    {"question": "How to read a file in Java?", "answer": "Use FileReader and BufferedReader: BufferedReader br = new BufferedReader(new FileReader('file.txt')); String line; while ((line = br.readLine()) != null) { // process line }"},
    {"question": "How to write a file in Java?", "answer": "Use FileWriter: FileWriter writer = new FileWriter('file.txt'); writer.write('Hello World'); writer.close();"},
    {"question": "What is a String in Java?", "answer": "A String is a sequence of characters."},
    {"question": "How to concatenate strings in Java?", "answer": "Use + operator: String result = str1 + str2;"},
    {"question": "What is a List in Java?", "answer": "A List is an ordered collection that can contain duplicate elements."},
    {"question": "How to create a List in Java?", "answer": "Use ArrayList: List<Type> list = new ArrayList<>();"},
    {"question": "What is a Map in Java?", "answer": "A Map is a collection of key-value pairs."},
    {"question": "How to create a Map in Java?", "answer": "Use HashMap: Map<KeyType, ValueType> map = new HashMap<>();"},
    {"question": "What is a Set in Java?", "answer": "A Set is a collection that cannot contain duplicate elements."},
    {"question": "How to create a Set in Java?", "answer": "Use HashSet: Set<Type> set = new HashSet<>();"},
    {"question": "What is a thread in Java?", "answer": "A thread is a lightweight process that can run concurrently with other threads."},
    {"question": "How to create a thread in Java?", "answer": "Use Thread class: Thread t = new Thread(() -> { // code }); t.start();"},
    {"question": "What is a Java Virtual Machine (JVM)?", "answer": "JVM is an engine that provides a runtime environment to execute Java bytecode."},
    {"question": "What is garbage collection in Java?", "answer": "Garbage collection is the process of automatically freeing memory by removing objects that are no longer in use."},
    {"question": "What is a package in Java?", "answer": "A package is a namespace that organizes a set of related classes and interfaces."},
    {"question": "How to import a package in Java?", "answer": "Use import statement: import package_name.ClassName;"},
    {"question": "What is a Java API?", "answer": "Java API is a set of classes and interfaces that provide functionality for Java applications."},
    {"question": "What is a framework in Java?", "answer": "A framework is a collection of libraries and tools that provide a structure for building applications."},
    {"question": "What is a framework in Python?", "answer": "A framework is a collection of libraries and tools that provide a structure for building applications."},
    {"question": "What is a NullPointerException in Java?", "answer": "It happens when you try to use an object reference that is null."},
    {"question": "How to install a Python package?", "answer": "Use pip like this: pip install package_name"},
    {"question": "What is JavaScript?", "answer": "JavaScript is a high-level, dynamic, untyped, and interpreted programming language."},
    {"question": "What is the use of JavaScript?", "answer": "JavaScript is used to create interactive effects within web browsers."},
    {"question": "How to include JavaScript in HTML?", "answer": "Use <script> tag: <script src='script.js'></script>"},
    {"question": "What is a variable in JavaScript?", "answer": "A variable is a container for storing data values."},
    {"question": "How do I declare a variable in JavaScript?", "answer": "Use let, const, or var: let x = 10;"},
    {"question": "What is a function in JavaScript?", "answer": "A function is a block of code designed to perform a particular task."},
    {"question": "How to create a function in JavaScript?", "answer": "Use function keyword: function myFunction() { // code }"},
    {"question": "What is an array in JavaScript?", "answer": "An array is a collection of items stored at contiguous memory locations."},
    {"question": "How to create an array in JavaScript?", "answer": "Use square brackets: let myArray = [1, 2, 3];"},
    {"question": "What is an object in JavaScript?", "answer": "An object is a standalone entity, with properties and type."},
    {"question": "How to create an object in JavaScript?", "answer": "Use curly braces: let myObject = { key: 'value' };"},
    {"question": "What is a promise in JavaScript?", "answer": "A promise is an object representing the eventual completion or failure of an asynchronous operation."},
    {"question": "How to handle promises in JavaScript?", "answer": ".then() for success and .catch() for error handling."},
    {"question": "What is an event in JavaScript?", "answer": "An event is an action that occurs in the browser, like a click or keypress."},
    {"question": "How to add an event listener in JavaScript?", "answer": "Use addEventListener: element.addEventListener('click', function() { // code });"},
    {"question": "What is the Document Object Model (DOM)?", "answer": "The DOM is a programming interface for web documents."},
    {"question": "How to manipulate the DOM in JavaScript?", "answer": "Use methods like getElementById, querySelector, etc."},
    {"question": "What is AJAX?", "answer": "AJAX (Asynchronous JavaScript and XML) allows web pages to be updated asynchronously."},
    {"question": "How to make an AJAX request in JavaScript?", "answer": "Use XMLHttpRequest or fetch API."},
    {"question": "What is JSON?", "answer": "JSON (JavaScript Object Notation) is a lightweight data interchange format."},
    {"question": "How to parse JSON in JavaScript?", "answer": "Use JSON.parse() method."},
    {"question": "How to stringify an object in JavaScript?", "answer": "Use JSON.stringify() method."},
    {"question": "What is a closure in JavaScript?", "answer": "A closure is a function that has access to its outer function's scope even after the outer function has returned."},
    {"question": "What is hoisting in JavaScript?", "answer": "Hoisting is JavaScript's default behavior of moving declarations to the top of the current scope."},
    {"question": "What is the difference between == and === in JavaScript?", "answer": "'==' checks for value equality, while '===' checks for both value and type equality."},
    {"question": "Bye", "answer": "Bye! Have a good day!"}
]


stop_words = set(stopwords.words('english'))


def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    return " ".join(tokens)


questions = [d['question'] for d in data]
processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_questions)

def detect_code_request(text):
    keywords = ["generate code", "write code", "code for", "create code", "give the", "give me", "example code"]
    return any(kw in text.lower() for kw in keywords)

def generate_code(task):
    task = task.lower()

    if "fibonacci" in task:
        return '''# Fibonacci sequence in Python\n\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        print(a, end=" ")\n        a, b = b, a + b\n\nfibonacci(10)'''

    elif "factorial" in task:
        return '''# Factorial using recursion\n\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)\n\nprint(factorial(5))'''

    elif "sum" in task:
        return '''# Sum of elements in a list\n\nnumbers = [1, 2, 3, 4, 5]\ntotal = sum(numbers)\nprint("Sum:", total)'''

    elif "sort" in task:
        return '''# Sort a list in Python\n\nnumbers = [5, 2, 9, 1, 3]\nnumbers.sort()\nprint("Sorted list:", numbers)'''

    elif "prime" in task:
        return '''# Check if a number is prime\n\ndef is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0:\n            return False\n    return True\n\nprint(is_prime(17))'''

    elif "reverse string" in task or "string reverse" in task:
        return '''# Reverse a string in Python\n\ndef reverse_string(s):\n    return s[::-1]\n\nprint(reverse_string("hello"))'''

    elif "palindrome" in task:
        return '''# Check if a string is a palindrome\n\ndef is_palindrome(s):\n    return s == s[::-1]\n\nprint(is_palindrome("madam"))'''

    elif "even or odd" in task or "odd or even" in task:
        return '''# Check if a number is even or odd\n\ndef even_or_odd(n):\n    if n % 2 == 0:\n        print("Even")\n    else:\n        print("Odd")\n\neven_or_odd(7)'''

    elif "largest" in task:
        return '''# Find the largest of three numbers\n\ndef find_largest(a, b, c):\n    return max(a, b, c)\n\nprint(find_largest(10, 20, 15))'''

    elif "leap year" in task:
        return '''# Check for leap year\n\ndef is_leap(year):\n    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):\n        return True\n    return False\n\nprint(is_leap(2024))'''

    elif "table" in task:
        return '''# Print multiplication table\n\ndef multiplication_table(n):\n    for i in range(1, 11):\n        print(f"{n} x {i} = {n*i}")\n\nmultiplication_table(5)'''

    elif "armstrong" in task:
        return '''# Armstrong number check\n\ndef is_armstrong(num):\n    digits = [int(d) for d in str(num)]\n    power = len(digits)\n    return sum(d ** power for d in digits) == num\n\nprint(is_armstrong(153))'''

    elif "swap" in task:
        return '''# Swap two numbers\n\na = 5\nb = 10\na, b = b, a\nprint("a =", a)\nprint("b =", b)'''

    elif "area of circle" in task:
        return '''# Calculate area of a circle\n\nimport math\nr = 7\narea = math.pi * r * r\nprint("Area of circle:", area)'''

    else:
        return "I can generate code for: Fibonacci, factorial, sum, prime, reverse string, palindrome, even/odd, sort list, leap year, table, Armstrong number, area of circle, and more."


def get_answer(user_input):
    if detect_code_request(user_input):
        return generate_code(user_input)
    else:
        user_input_processed = preprocess(user_input)
        user_vec = vectorizer.transform([user_input_processed])
        similarity = cosine_similarity(user_vec, X)
        max_score = similarity.max()

        if max_score < 0.3:
            return "I'm not sure I understand. Can you rephrase that?"

        index = similarity.argmax()
        return data[index]['answer']


def send_message(event=None): 
    user_input = entry.get()
    if user_input.strip() == "":
        return

    chat_window.insert(tk.END, "You: " + user_input + "\n\n", "user")
    entry.delete(0, tk.END)

    response = get_answer(user_input)
    chat_window.insert(tk.END, "Bot: " + response + "\n\n", "bot")
    chat_window.see(tk.END)

root = tk.Tk()
root.title("Coding Chatbot")
root.geometry("800x700")
root.configure(bg="white")

title = tk.Label(root, text="Coding Chatbot", font=("Arial", 20, "bold"), bg="white")
title.pack(pady=10)

chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 12), bg="#f0f0f0", padx=10, pady=10)
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

chat_window.tag_configure("user", justify='right', foreground="blue")
chat_window.tag_configure("bot", justify='left', foreground="black")

chat_window.insert(tk.END, "Bot: How can I help you today?\n\n", "bot")

input_frame = tk.Frame(root, bg="white")
input_frame.pack(pady=10, padx=10, fill=tk.X)

entry = tk.Entry(input_frame, font=("Arial", 12), width=50)
entry.pack(side=tk.LEFT, padx=(0, 10), ipady=6, expand=True, fill=tk.X)
entry.bind("<Return>", send_message)

send_button = tk.Button(input_frame, text="↑", font=("Arial", 14), width=4, command=send_message)
send_button.pack(side=tk.RIGHT)

root.mainloop()
