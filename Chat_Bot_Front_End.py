import dearpygui.dearpygui as dpg
import Chat_Bot_Back_End


def makeAllComponents(input_text_theme, text_color, background_color, message_window_color, default_font):

    
    with dpg.window(tag="Background", label="Background",
                    no_title_bar=True, no_resize=True, no_move=True,
                    no_collapse=True, no_scrollbar=True,
                    no_close=True, no_bring_to_front_on_focus=True):
        

            with dpg.child_window(tag="Message_Box", width=540, height=540, border=True,
                                autosize_y=False, autosize_x=False):
                chat_bot_messages = dpg.add_text("How can I help you today?", wrap=500)
                dpg.bind_item_theme(chat_bot_messages, text_color)

           
            input_box = dpg.add_input_text(tag="user_input", on_enter=True, callback=Chat_Bot_Back_End.handle_enter)
            dpg.bind_item_theme(input_box, input_text_theme)
            dpg.set_item_pos(input_box, [30, 590]) 
            dpg.set_item_height(input_box, 30)
            dpg.set_item_width(input_box, 500)


    
    dpg.bind_item_theme("Background", background_color)
    dpg.bind_item_theme("Message_Box", message_window_color)

    
    dpg.set_item_pos("Message_Box", [20, 30])
    dpg.set_item_width("Background", dpg.get_viewport_width())
    dpg.set_item_height("Background", dpg.get_viewport_height())
    dpg.set_item_pos("Background", [0, 0])

    
    
    dpg.bind_font(default_font)
    dpg.setup_dearpygui()
    dpg.set_primary_window("Background", True)
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

def main():

    dpg.create_context()
    dpg.create_viewport(title="Dumb Chat Bot", width=600, height=700)



        
    with dpg.font_registry():
        default_font = dpg.add_font("BitcountSingle-VariableFont_CRSV,ELSH,ELXP,slnt,wght.ttf", 20)

    
    with dpg.theme() as background_color:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (100, 150, 250, 255))  

    
    with dpg.theme() as message_window_color:
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (255, 255, 255, 255)) 
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10.0) 

    
    with dpg.theme() as input_text_theme:
        with dpg.theme_component(dpg.mvInputText):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (255, 255, 255, 255))      
            dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0, 255))               
            dpg.add_theme_color(dpg.mvThemeCol_Border, (128, 128, 128, 255))       
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1.0)               
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10.0)  

    with dpg.theme() as text_color:
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (30, 30, 30, 255))  # dark gray



    makeAllComponents(input_text_theme, text_color, background_color, message_window_color, default_font)


if __name__ == "__main__":
    main()
