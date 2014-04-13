Want pretty syntax highlighting when editing your .cu files in Visual Studio?
Heres how:

---
Visual Studio .Net / 7:

1. If you don't have a usertype.dat file in your Microsoft Visual Studio .Net\Common7\IDE folder, then copy the included usertype.dat file there.  If you do, append the contents of the included usertype.dat onto the end of the .Net\Common7\IDE\usertype.dat

2. Open up the registry editor and go to the following location - HKLM\SOFTWARE\Microsoft\VisualStudio\7.1\Languages\File Extensions.

3. Copy the default value from the .cpp key.

4. Create a new key under the File Extensions with the name of .cu

5. Paste the value you just copied into the default value

6. Restart Visual Studio and your CUDA code should now have syntax highlighting

NOTE: You can use the install_cuda_highlighting_vs7.reg file to simplify the above steps 2-5.  Simply double-click on the file and press OK when prompted.
