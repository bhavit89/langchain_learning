


first_func = lambda x : f"word_count :{len(x.split())}\n{x}"

input_str = "Mercedes is a competitor of BMW"
output = first_func(input_str)

second_func = lambda x : {"text":output , "language":"French"}
print(output)


result = second_func(output)
print(result)