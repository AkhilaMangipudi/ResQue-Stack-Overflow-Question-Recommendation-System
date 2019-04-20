from tkinter import *
# Driver code

def clear():
    # clear the content of text entry box
    name_field.delete(0, END)
    course_field.delete(0, END)


if __name__ == "__main__":
    root = Tk()

    # set the background colour of GUI window
    root.configure(background='light gray')

    # set the title of GUI window
    root.title("registration form")

    # set the configuration of GUI window
    root.geometry("500x300")
    heading = Label(root, text="Form", bg="light gray")

    # create a Name label
    name = Label(root, text="Enter Question Body", bg="light gray")

    # create a Course label
    course = Label(root, text="Enter Question Title", bg="light gray")
    heading.grid(row=0, column=1)
    name.grid(row=1, column=0)
    course.grid(row=2, column=0)

    name_field = Entry(root)
    course_field = Entry(root)

    name_field.grid(row=1, column=1, ipadx="100")
    course_field.grid(row=2, column=1, ipadx="100")

    submit = Button(root, text="Recommend", fg="Black",
                    bg="Gray")
    submit.grid(row=8, column=1)

    root.mainloop()
