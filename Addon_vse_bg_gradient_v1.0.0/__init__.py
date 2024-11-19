bl_info = {
    "name": "VSE BG Gradient",
    "description": "Custom multi gradient-color, Generate text effects typewriter.",
    "author": "Dinesh",
    "version": (1, 0, 0),
    "blender": (2, 8, 0),
    "category": "Sequencer",
    "location": "Sequencer > ToolTab",
}
import bpy
from bpy.props import FloatVectorProperty, FloatProperty, EnumProperty, CollectionProperty, IntProperty
import os
import math
import numpy as np

# Enum for Gradient Types
GRADIENT_TYPES = [
    ('LINEAR', 'Linear', 'Linear gradient'),
    ('RADIAL', 'Radial', 'Radial gradient'),
    ('ANGULAR', 'Angular', 'Angular gradient'),
    ('DIAMOND', 'Diamond', 'Diamond gradient'),
    ('SPIRAL', 'Spiral', 'Spiral gradient'),
    ('CHECKERBOARD', 'Checkerboard', 'Checkerboard gradient')

]
gradient_counters = {
    "Linear": 0,
    "Radial": 0,
    "Angular": 0,
    "Diamond": 0,
    "Spiral": 0,
}

def generate_unique_filename(gradient_type, extension="png"):

    global gradient_counters
    
    # Increment the counter for the gradient type
    if gradient_type in gradient_counters:
        gradient_counters[gradient_type] += 1
    else:
        gradient_counters[gradient_type] = 1  # Initialize if not present
    
    # Generate the file name
    counter = gradient_counters[gradient_type]
    return f"{gradient_type}_Gradient_{counter}.{extension}"

# Function to create a gradient image and add it to the VSE
def create_gradient_image(context):
    gradient_props = context.scene.gradient_props
    width, height = context.scene.render.resolution_x, context.scene.render.resolution_y
    angle = math.radians(gradient_props.gradient_rotation)
    gradient_type = gradient_props.gradient_type
    scale = gradient_props.gradient_scale
    location = (gradient_props.location_x, gradient_props.location_y)
    interpolation = gradient_props.gradient_interpolation
    repeat_mode = gradient_props.repeat_mode
    flip = gradient_props.gradient_color_flip

    # Generate a unique file name
    file_name = generate_unique_filename(gradient_type)
    file_path = os.path.join(bpy.app.tempdir, file_name)

    if file_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[file_name])

    image = bpy.data.images.new(name=file_name, width=width, height=height)
    
    # Use numpy for efficient pixel manipulation
    pixels = np.zeros((height, width, 4), dtype=np.float32)  # RGBA

    # Custom gradient colors array
    colors = [color.color for color in gradient_props.custom_colors]
    
    # Create gradient based on the selected type
    if gradient_type == 'LINEAR':
        generate_linear_gradient(pixels, width, height, colors, angle, scale, location, interpolation, repeat_mode, flip)
    elif gradient_type == 'RADIAL':
        generate_radial_gradient(pixels, width, height, colors, scale, location, interpolation, repeat_mode, flip)
    elif gradient_type == 'ANGULAR':
        generate_angular_gradient(pixels, width, height, colors, angle, location, scale, interpolation, repeat_mode, flip)
    elif gradient_type == 'DIAMOND':
        generate_diamond_gradient(pixels, width, height, colors, angle, scale, location, interpolation, repeat_mode, flip)
    elif gradient_type == 'SPIRAL':
        generate_spiral_gradient(pixels, width, height, colors, angle, scale, location, interpolation, repeat_mode, flip)
    elif gradient_type == 'CHECKERBOARD':
        generate_checkerboard_gradient(pixels, width, height, colors, angle, scale)

    image.pixels = pixels.flatten()
    image.filepath_raw = file_path
    image.file_format = 'PNG'
    image.save()

    return file_path

def handle_repeat_mode(t, repeat_mode):

    if repeat_mode == 'NONE':
        # Clamp `t` between 0 and 1
        return max(0.0, min(1.0, t))
    elif repeat_mode == 'DIRECT':
        # Repeat `t` in cycles of [0, 1]
        return t % 1.0
    elif repeat_mode == 'REFLECTED':
        # Reflect `t` back and forth between [0, 1]
        cycle = math.floor(t)
        return 1.0 - (t % 1.0) if cycle % 2 == 1 else t % 1.0
    else:
        return t  # Default fallback
    

def blend_b_spline(t, colors):

    total_colors = len(colors)
    scaled_t = t * (total_colors - 1)
    index = int(scaled_t)

    # Wrap indices to handle boundaries
    p0 = colors[max(0, index - 1)]
    p1 = colors[max(0, index)]
    p2 = colors[min(total_colors - 1, index + 1)]
    p3 = colors[min(total_colors - 1, index + 2)]

    # Compute fractional position within the current interval
    t_blend = scaled_t - index

    # B-spline basis function weights
    b0 = (1 - t_blend) ** 3 / 6.0
    b1 = (3 * t_blend ** 3 - 6 * t_blend ** 2 + 4) / 6.0
    b2 = (-3 * t_blend ** 3 + 3 * t_blend ** 2 + 3 * t_blend + 1) / 6.0
    b3 = t_blend ** 3 / 6.0

    return [
        b0 * p0[i] + b1 * p1[i] + b2 * p2[i] + b3 * p3[i] for i in range(4)
    ]


# next Interpolation
def next_interpolation(t):
    return 0.5 - 0.5 * np.sin(t * np.pi)

def blend_custom_colors(t, colors, interpolation='LINEAR'):
    "Interpolates across multiple colors based on factor t (0 to 1)."
    total_colors = len(colors)
    if total_colors == 0:
        return (0, 0, 0, 1)
    
    # Scale t to the range of colors
    scaled_t = t * (total_colors - 1)
    index = int(scaled_t)
    t_blend = scaled_t - index

    # Blend between two colors based on scaled position
    if index < total_colors - 1:
        color1 = colors[index]
        color2 = colors[index + 1]
    else:
        color1 = colors[-1]
        color2 = colors[-1]

    # Perform interpolation
    if interpolation == 'LINEAR':
        return [(1 - t_blend) * color1[i] + t_blend * color2[i] for i in range(4)]
    elif interpolation == 'EASE':
        t_blend = t_blend * t_blend * (3 - 2 * t_blend)  # Smoothstep formula
        return [(1 - t_blend) * color1[i] + t_blend * color2[i] for i in range(4)]
    elif interpolation == 'CONSTANT':
        return color1
    elif interpolation == 'B_SPLINE':
        return blend_b_spline(t, colors)
    elif interpolation == 'NEXT_BLEND':
        return [(1 - next_interpolation(t_blend)) * color1[i] + next_interpolation(t_blend) * color2[i] for i in range(4)]

    return [
        (1 - t_blend) * color1[i] + t_blend * color2[i] for i in range(4)
    ]

def generate_linear_gradient(pixels, width, height, colors, angle, scale, interpolation='LINEAR', repeat_mode='NONE', flip=False):

    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)

    colors = colors[::-1] if flip else colors

    for y in range(height):
        for x in range(width):
            # Normalize coordinates to [-1, 1] range
            nx, ny = ((x / width) * 2 - 1), ((y / height) * 2 - 1)
            # Rotate coordinates by the given angle
            rx = (cos_theta * nx - sin_theta * ny) / scale
            # Map rx to [0, 1] range
            t = (rx + 1) / 2
            t = handle_repeat_mode(t, repeat_mode)
            pixels[y, x] = blend_custom_colors(t, colors, interpolation)

def generate_radial_gradient(pixels, width, height, colors, scale, interpolation='LINEAR', repeat_mode='NONE', flip=False):
    cx, cy = width / 2, height / 2
    max_dist = math.sqrt(cx**2 + cy**2) * scale

    colors = colors[::-1] if flip else colors

    for y in range(height):
        for x in range(width):
            # Apply scale to the radial distance
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist
            dist = handle_repeat_mode(dist, repeat_mode)
            pixels[y, x] = blend_custom_colors(dist, colors, interpolation)

def generate_angular_gradient(pixels, width, height, colors, angle, scale, interpolation='LINEAR', repeat_mode='NONE', flip=False):

    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)

    colors = colors[::-1] if flip else colors

    # Calculate center point of the image
    cx, cy = width / 2, height / 2

    for y in range(height):
        for x in range(width):
            # Calculate the distance and angle from the center of the image
            dx, dy = x - cx, y - cy
            distance = math.sqrt(dx**2 + dy**2)  # Distance from center
            raw_angle = math.atan2(dy, dx)  # Angle of the current pixel from the center

            normalized_angle = (raw_angle + math.pi) / (2 * math.pi)

            t = normalized_angle * scale
            t = handle_repeat_mode(t, repeat_mode)
            pixels[y, x] = blend_custom_colors(t, colors, interpolation)

def generate_diamond_gradient(pixels, width, height, colors, angle, scale, interpolation='LINEAR', repeat_mode='NONE', flip=False):
 
    cx, cy = width / 2, height / 2  # Center of the image
    max_dist = (cx + cy) * scale  # Max distance scaled by scale factor

    colors = colors[::-1] if flip else colors

    angle = math.radians(angle)

    for y in range(height):
        for x in range(width):
            # Apply rotation to the coordinates
            dx, dy = x - cx, y - cy
            # Rotate the point (x, y) by the angle
            rx = math.cos(angle) * dx - math.sin(angle) * dy
            ry = math.sin(angle) * dx + math.cos(angle) * dy

            dist = (abs(rx) + abs(ry)) / max_dist
            dist = handle_repeat_mode(dist, repeat_mode)
            pixels[y, x] = blend_custom_colors(min(dist, 1.0), colors, interpolation)

def generate_spiral_gradient(pixels, width, height, colors, angle, scale, interpolation='LINEAR', repeat_mode='NONE', flip=False):

    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    
    # Flip colors if needed
    colors = colors[::-1] if flip else colors
    
    cx, cy = width / 2, height / 2
    max_dist = math.sqrt(cx * cx + cy * cy)  # Maximum distance from the center
    
    for y in range(height):
        for x in range(width):
            dx, dy = x - cx, y - cy
            dist = math.sqrt(dx ** 2 + dy ** 2) / max_dist  # Normalize distance to [0, 1]
            
            # Calculate the angle for the spiral pattern
            spiral_angle = math.atan2(dy, dx) + dist * scale * math.pi * 2  # Spiral effect
            
            # Normalize angle within the range [0, 1] based on a full revolution (2π)
            t = (spiral_angle % (2 * math.pi)) / (2 * math.pi)
            
            # Apply repeat_mode logic
            if repeat_mode == 'DIRECT':
                t = t % 1  # Direct repeating without reflecting
            elif repeat_mode == 'REFLECTED':
                t = 1 - abs((t * 2) % 2 - 1)  # Reflected effect (bounces back after each loop)
            elif repeat_mode == 'NONE':
                #t = min(max(t, 1), 1)
                t = max(0, min(t, 1)) 

            pixels[y, x] = blend_custom_colors(t, colors, interpolation)

def generate_checkerboard_gradient(pixels, width, height, colors, angle, scale):

    if not colors or len(colors) < 2:
        colors = [(0, 0, 0, 1), (1, 1, 1, 1)]  # Default to black and white

    num_colors = len(colors)
    cell_size = max(1, scale * 10)  # Ensure non-zero cell size
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)

    for y in range(height):
        for x in range(width):
            # Rotate coordinates based on the angle
            nx = cos_theta * x - sin_theta * y
            ny = sin_theta * x + cos_theta * y

            # Determine cell indices
            cell_x = int(nx // cell_size)
            cell_y = int(ny // cell_size)

            # Determine gradient mix factor within the cell
            local_x = (nx % cell_size) / cell_size
            local_y = (ny % cell_size) / cell_size

            # Choose gradient direction and blending
            gradient_t = (local_x + local_y) / 2  # Diagonal gradient in the box

            # Alternate between two colors based on the checkerboard index
            base_index = (cell_x + cell_y) % num_colors
            next_index = (base_index + 1) % num_colors

            # Blend between the two colors
            pixels[y, x] = blend_custom_colors(gradient_t, [colors[base_index], colors[next_index]])

# Property for custom colors
class CustomColorProperty(bpy.types.PropertyGroup):
        color: FloatVectorProperty(
        name="Color",
        subtype='COLOR',
        size=4,
        min=0.0, max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),
        description="Custom color for gradient"
    )

# Operator to create and add the gradient image as an IMAGE strip in the VSE
class VSE_OT_AddGradientImage(bpy.types.Operator):
    bl_idname = "vse.add_gradient_image"
    bl_label = "Add Gradient Image"
    bl_description = "Add custom gradient image with multiple colors as an IMAGE strip to the VSE"

    def execute(self, context):
        gradient_props = context.scene.gradient_props
        filepath = create_gradient_image(context)
        frame_start = context.scene.frame_current
        frame_end = frame_start + gradient_props.gradient_duration - 1

        # Add image strip with custom duration
        bpy.ops.sequencer.image_strip_add(
            directory=os.path.dirname(filepath),
            files=[{"name": os.path.basename(filepath)}],
            frame_start=frame_start,
            frame_end=frame_end
        )

        file_name = os.path.basename(filepath)
        self.report({'INFO'}, f"Gradient image '{file_name}' added successfully")
        return {'FINISHED'}

# Define properties for the gradient panel
class VSEGradientPanelProperties(bpy.types.PropertyGroup):
    custom_colors: CollectionProperty(type=CustomColorProperty)
    gradient_rotation: FloatProperty(
        name="Rotation Angle",
        description="Rotation angle of the gradient in degrees",
        default=0.0,
    )
    gradient_type: EnumProperty(
        name="Gradient Type",
        description="Choose the type of gradient",
        items=GRADIENT_TYPES,
        default='LINEAR',
    )
    gradient_scale: FloatProperty(
        name="Gradient Scale",
        default=1.0,
        description="Scale of the gradient"
    )
    gradient_duration: IntProperty(
        name="Duration",
        description="Duration of the gradient image strip in frames",
        default=24,
        min=1,
    )
    gradient_interpolation: EnumProperty(
        name="Interpolation",
        description="Interpolation mode for gradient blending",
        items=[
            ('LINEAR', "Linear", "Linear interpolation"),
            ('EASE', "Ease", "Smooth ease-in-out interpolation"),
            ('B_SPLINE', "B-Spline", "Smooth curve interpolation"),
            ('CONSTANT', "Constant", "No interpolation between colors"),
            ('NEXT_BLEND', "Next-Blend", "Smooth next solor blend transitions for oscillating gradients")
        ],
        default='LINEAR',
    )
    repeat_mode: EnumProperty(
        name="Repeat Mode",
        description="Repetition mode for gradients",
        items=[
            ('NONE', "None", "No repetition"),
            ('DIRECT', "Direct", "Repeat gradient directly"),
            ('REFLECTED', "Reflected", "Alternate gradient direction")
        ],
        default='NONE',
    )
    gradient_color_flip: bpy.props.BoolProperty(
        name="Flip Colors",
        description="Reverse the order of colors in the gradient",
        default=False,
    ) 


# UI Panel to control gradient properties
class VSE_PT_GradientPanel(bpy.types.Panel):
    bl_label = "Multi-Color Custom Gradient"
    bl_idname = "VSE_PT_gradient_panel"
    bl_space_type = 'SEQUENCE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Tool'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.gradient_props

        layout.prop(props, "gradient_type", text="Gradient Type", icon='NODE_TEXTURE')
        layout.prop(props, "gradient_interpolation", text="Interpolation")
        layout.prop(props, "repeat_mode", text="Repeat Mode")
        
        layout.prop(props, "gradient_duration", text="Duration")
        layout.prop(props, "gradient_scale", text="Gradient Scale")
        layout.prop(props, "gradient_rotation", text="Rotation Angle")

        row = layout.row()
        row.label(text="Custom Colors:")
        row.prop(props, "gradient_color_flip", text="", icon='ARROW_LEFTRIGHT', toggle=True)
        row.operator("vse.add_custom_color", icon='ADD', text="Add Color")

        for i, color in enumerate(props.custom_colors):
            row = layout.row() 
            row.prop(color, "color", text=f"Color {i+1}")
            row.operator("vse.remove_custom_color", icon='REMOVE', text="").index = i
    
        row = layout.row()
        row.operator("vse.add_gradient_image", text="Add Gradient Image")

# Operators for adding and removing custom colors
class VSE_OT_AddCustomColor(bpy.types.Operator):
    bl_idname = "vse.add_custom_color"
    bl_label = "Add Custom Color"
    bl_description = "Add a new color stop to the gradient"

    def execute(self, context):
        context.scene.gradient_props.custom_colors.add()
        return {'FINISHED'}

class VSE_OT_RemoveCustomColor(bpy.types.Operator):
    bl_idname = "vse.remove_custom_color"
    bl_label = "Remove Custom Color"
    bl_description = "Remove selected color stop from the gradient"

    index: IntProperty()

    def execute(self, context):
        custom_colors = context.scene.gradient_props.custom_colors
        if 0 <= self.index < len(custom_colors):
            custom_colors.remove(self.index)
        return {'FINISHED'}

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def apply_typewriter_effect(scene, text_strip, duration, align_x, align_y, split_mode, location_x, location_y):
    """Generate text effects per letter or word while retaining original styles."""
    text = text_strip.text
    split_items = text.split(" ") if split_mode == 'WORD' else list(text)
    frame_start = text_strip.frame_start
    strip_channel = text_strip.channel

    # Mute the original strip
    text_strip.mute = True
    created_strips = []

    for i, item in enumerate(split_items):
        bpy.ops.sequencer.select_all(action='DESELECT')

        # Create a new text strip
        bpy.ops.sequencer.effect_strip_add(
            type='TEXT', 
            frame_start=int(frame_start + i * (duration // len(split_items))),
            frame_end=int(frame_start + (i + 1) * (duration // len(split_items))),
            channel=strip_channel + 1
        )

        new_strip = scene.sequence_editor.active_strip
        new_strip.text = "".join(split_items[:i + 1]) if split_mode == 'LETTER' else " ".join(split_items[:i + 1])

        # Copy text style properties
        for attr in ["wrap_width", "use_bold", "use_italic", "font_size", "color", "use_shadow", "shadow_color", "shadow_angle", "shadow_blur", "use_outline", "outline_color", "outline_width", "use_box", "box_color", "box_margin", "blend_alpha"]:
            setattr(new_strip, attr, getattr(text_strip, attr))

        # font data
        new_strip.font = text_strip.font

        # Set alignment
        new_strip.align_x = align_x
        new_strip.align_y = align_y

        # Set location using operator (location_x and location_y are passed in)
        new_strip.location = (location_x, location_y)

        # Set unique name for each new strip
        new_strip.name = f"Text_{i + 1}"

        created_strips.append(new_strip)

    # Group created strips into a meta strip
    for strip in created_strips:
        strip.select = True

    bpy.ops.sequencer.meta_make()
    # Name the meta strip
    meta_strip = scene.sequence_editor.active_strip
    # Get the current count of meta strips in the scene
    meta_count = len([strip for strip in scene.sequence_editor.sequences_all if strip.type == 'META'])
    meta_strip.name = f"Typewriter_{meta_count}"

class VSE_OT_AddTextEffect(bpy.types.Operator):
    """Apply Text Effect"""
    bl_idname = "sequencer.add_text_effect"
    bl_label = "Typewriter Effect"
    bl_options = {'REGISTER', 'UNDO'}

    effect_type: bpy.props.EnumProperty(
        name="Effect",
        description="Select text effect to apply",
        items=[
            ('TYPEWRITER', "Typewriter", ""),
        ],
        default='TYPEWRITER',
    )
    duration: bpy.props.IntProperty(
        name="Duration",
        description="Duration of the effect",
        default=24,
        min=1
    )
    split_mode: bpy.props.EnumProperty(
        name="Split Mode",
        description="Apply effect per letter or per word",
        items=[
            ('LETTER', "Letter", "Generate effects per letter"),
            ('WORD', "Word", "Generate effects per word"),
        ],
        default='LETTER',
    )
    align_x: bpy.props.EnumProperty(
        name="Horizontal Alignment",
        description="Text horizontal alignment",
        items=[
            ('LEFT', "Left", "Align text to the left"),
            ('CENTER', "Center", "Center-align text"),
            ('RIGHT', "Right", "Align text to the right"),
        ],
        default='CENTER',
    )
    align_y: bpy.props.EnumProperty(
        name="Vertical Alignment",
        description="Text vertical alignment",
        items=[
            ('TOP', "Top", "Align text to the top"),
            ('CENTER', "Center", "Center-align text"),
            ('BOTTOM', "Bottom", "Align text to the bottom"),
        ],
        default='CENTER',
    )
    location_x: bpy.props.FloatProperty(
        name="Location X",
        description="Horizontal position of the text",
        default=0.5,  # Centered
        min=0.0,
        max=1.0
    )
    location_y: bpy.props.FloatProperty(
        name="Location Y",
        description="Vertical position of the text",
        default=0.5,  # Centered
        min=0.0,
        max=1.0
    )

    def execute(self, context):
        strip = context.scene.sequence_editor.active_strip
        if not strip or strip.type != 'TEXT':
            self.report({'WARNING'}, "Please select a text strip.")
            return {'CANCELLED'}

        if self.effect_type == 'TYPEWRITER':
            apply_typewriter_effect(
                context.scene,
                strip,
                self.duration,
                self.align_x,
                self.align_y,
                self.split_mode,
                self.location_x,
                self.location_y
            )

        return {'FINISHED'}

class VSE_PT_TextEffectPanel(bpy.types.Panel):
    bl_label = "Typewriter Effect"
    bl_idname = "VSE_PT_TextEffectPanel"
    bl_space_type = 'SEQUENCE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Tool"

    def draw(self, context):
        layout = self.layout
        layout.operator("sequencer.add_text_effect", text="Apply Effect", icon='FILE_TEXT')

# Registration
classes = [
    CustomColorProperty,
    VSEGradientPanelProperties,
    VSE_OT_AddGradientImage,
    VSE_PT_GradientPanel,
    VSE_OT_AddCustomColor,
    VSE_OT_RemoveCustomColor,
    VSE_OT_AddTextEffect,
    VSE_PT_TextEffectPanel
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.gradient_props = bpy.props.PointerProperty(type=VSEGradientPanelProperties)
    bpy.types.Scene.text_effect_split_mode = bpy.props.EnumProperty(
    name="Split Mode",
    description="Split effects per letter or per word",
    items=[('LETTER', "Letter", "Generate effects per letter"),
            ('WORD', "Word", "Generate effects per word")],
    default='LETTER',
    )
    bpy.types.Scene.text_effect_location_x = bpy.props.FloatProperty(
    name="Location X",
    description="Horizontal position of the text",
    default=0.5,  # Centered
    min=0.0,
    max=1.0
    )
    bpy.types.Scene.text_effect_location_y = bpy.props.FloatProperty(
    name="Location Y",
    description="Vertical position of the text",
    default=0.5,  # Centered
    min=0.0,
    max=1.0
    )

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.gradient_props 
    del bpy.types.Scene.text_effect_split_mode
    del bpy.types.Scene.text_effect_location_x
    del bpy.types.Scene.text_effect_location_y

if __name__ == "__main__":
    register()