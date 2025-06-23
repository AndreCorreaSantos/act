import glfw

if not glfw.init():
    print("Failed to initialize GLFW")
    exit(1)

window = glfw.create_window(640, 480, "Test Window", None, None)
if not window:
    glfw.terminate()
    print("Failed to create window")
    exit(1)

glfw.make_context_current(window)
print("GLFW context created successfully")

glfw.terminate()
