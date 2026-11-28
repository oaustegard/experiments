ENTRY(_start)
SECTIONS
{
    . = LOAD_ADDR;
    _start_image = .;
    .text : { KEEP(*(.text.boot)) *(.text .text.*) }
    .rodata : ALIGN(16) { *(.rodata .rodata.*) }
    .data : ALIGN(16) { *(.data .data.*) }
    .bss (NOLOAD) : ALIGN(16) {
        __bss_start = .;
        *(.bss .bss.*) *(COMMON)
        . = ALIGN(16);
        __bss_end = .;
    }
    . = ALIGN(4096);
    _end = .;
    _image_size = _end - _start_image;
    /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
