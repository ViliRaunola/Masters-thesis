import analyzer.reddit as reddit
import utility.common as common
import utility.menu as menu


def main():

    print("Hello World!")

    # Let user train the models
    while True:
        if common.check_model_folders():
            break

        userinput = menu.start_menu()
        menu.switch_start(userinput)

    nlp_tools = common.load_pipelines()
    prawn_connection = reddit.create_praw_instance()

    # Let user to use the main functionalities
    while True:
        userinput = menu.main_menu()
        menu.switch_main(
            input=userinput, prawn_connection=prawn_connection, nlp_tools=nlp_tools
        )


if __name__ == "__main__":
    main()
