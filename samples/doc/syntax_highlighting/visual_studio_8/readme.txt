Want pretty syntax highlighting when editing your .cu files in Visual Studio?
Here's how:

---
Visual Studio .Net 2005 / Visual Studio 8:

1. If you don't have a usertype.dat file in your "Microsoft Visual Studio 8\Common7\IDE" folder, then copy the included usertype.dat file there.  If you do, append the contents of the included usertype.dat onto the end of the "Microsoft Visual Studio 8\Common7\IDE\usertype.dat"

2. Start Visual Studio 8.  Select the menu "Tools->Options...".  Open "Text Editor" in the tree view on the left, and click on "File Extension".  Type cu in the "Extension" box, set the editor to "Microsoft Visual C++" and click "Add".  Click "OK" on the dialog box.  

3. Restart Visual Studio and your CUDA code should now have syntax highlighting.