initSidebarItems({"struct":[["DynamicState","The dynamic state to use for a draw command."],["PrimaryCommandBuffer","Represents a collection of commands to be executed by the GPU.A primary command buffer can contain any command."],["PrimaryCommandBufferBuilder","A prototype of a primary command buffer.Usage"],["PrimaryCommandBufferBuilderInlineDraw","Object that you obtain when calling `draw_inline` or `next_subpass_inline`."],["PrimaryCommandBufferBuilderSecondaryDraw","Object that you obtain when calling `draw_secondary` or `next_subpass_secondary`."],["SecondaryComputeCommandBuffer","Represents a collection of commands to be executed by the GPU.A secondary compute command buffer contains non-draw commands (like copy commands, compute shader execution, etc.). It can only be called outside of a renderpass."],["SecondaryComputeCommandBufferBuilder","A prototype of a secondary compute command buffer."],["SecondaryGraphicsCommandBuffer","Represents a collection of commands to be executed by the GPU.A secondary graphics command buffer contains draw commands and non-draw commands. Secondary command buffers can't specify which framebuffer they are drawing to. Instead you must create a primary command buffer, specify a framebuffer, and then call the secondary command buffer.A secondary graphics command buffer can't be called outside of a renderpass."],["SecondaryGraphicsCommandBufferBuilder","A prototype of a secondary compute command buffer."]]});