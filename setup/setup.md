# Dell xps 15 laptop setup

## Steps

1. install chrome
2. install atom
3. look into dual booting with kubuntu
4. set up python and jekyll with Bash on Ubuntu on Windows

### Set up python with Bash on Ubuntu on Windows
Following this [guide](http://timmyreilly.azurewebsites.net/python-with-ubuntu-on-windows/)



### Dual booting
- [x] find a guide online
  - https://hackernoon.com/installing-ubuntu-18-04-along-with-windows-10-dual-boot-installation-for-deep-learning-f4cd91b58557
  - https://askubuntu.com/questions/1031993/how-to-install-ubuntu-18-04-alongside-windows-10
  - https://userbase.kde.org/Kubuntu/Installation
- [x] make windows recovery drive
  - [x] get 16 gb usb drive
- [x] download kubuntu
  - https://kubuntu.org/alternative-downloads
  - kubuntu-19.04-desktop-amd64.iso
8e43da4ddba84e1e67036aac053ba32079e6fb81a28aaedae8a8e559ac1a4d3f
  - check
8e43da4ddba84e1e67036aac053ba32079e6fb81a28aaedae8a8e559ac1a4d3f
- make partition for kubuntu
  - 50 gb, gpt
- make kubuntu bootable live usb
  - using https://unetbootin.github.io/
  - using rufus
- kubuntu doesnt seem to be able to install on disk, stuck at size issue (8 gb needed)
- ubuntu-18.04.2-desktop-amd64.iso
22580b9f3b186cc66818e60f44c46f795d708a1ad86b9225c458413b638459c4
- check
22580b9f3b186cc66818e60f44c46f795d708a1ad86b9225c458413b638459c4
- 050819: having problems even starting live version of kubuntu or Ubuntu
  - think it is video driver/nouveau issue
  - now following:
    - https://medium.com/@pwaterz/how-to-dual-boot-windows-10-and-ubuntu-18-04-on-the-15-inch-dell-xps-9570-with-nvidia-1050ti-gpu-4b9a2901493d
    - https://github.com/rcasero/doc/wiki/Ubuntu-linux-on-Dell-XPS-15-%289560%29
    - may need to change nouveau boot parameter https://github.com/rcasero/doc/issues/6
    - for changing boot parameters https://help.ubuntu.com/community/BootOptions
- UEFI issues
  - https://help.ubuntu.com/community/Installation/FromUSBStick#UEFI
  - https://help.ubuntu.com/community/UEFI
  - https://ubuntuforums.org/showthread.php?t=2147295
