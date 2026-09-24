from graph import graph


print("\nSpecialty Pharmacy AI Agent")
print("Type 'exit' to stop.\n")


while True:
    question = input("Ask a question: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    result = graph.invoke(
        {
            "question": question,
            "answer": ""
        }
    )

    print("\nAnswer:")
    print(result["answer"])
    print()