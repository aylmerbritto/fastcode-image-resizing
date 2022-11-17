
./build/kernel:     file format elf64-x86-64


Disassembly of section .init:

00000000004016d0 <_init>:
  4016d0:	48 83 ec 08          	sub    $0x8,%rsp
  4016d4:	48 8b 05 1d 79 20 00 	mov    0x20791d(%rip),%rax        # 608ff8 <__gmon_start__>
  4016db:	48 85 c0             	test   %rax,%rax
  4016de:	74 05                	je     4016e5 <_init+0x15>
  4016e0:	e8 cb 00 00 00       	callq  4017b0 <__gmon_start__@plt>
  4016e5:	48 83 c4 08          	add    $0x8,%rsp
  4016e9:	c3                   	retq   

Disassembly of section .plt:

00000000004016f0 <.plt>:
  4016f0:	ff 35 12 79 20 00    	pushq  0x207912(%rip)        # 609008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4016f6:	ff 25 14 79 20 00    	jmpq   *0x207914(%rip)        # 609010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4016fc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401700 <__cxa_begin_catch@plt>:
  401700:	ff 25 12 79 20 00    	jmpq   *0x207912(%rip)        # 609018 <__cxa_begin_catch@CXXABI_1.3>
  401706:	68 00 00 00 00       	pushq  $0x0
  40170b:	e9 e0 ff ff ff       	jmpq   4016f0 <.plt>

0000000000401710 <printf@plt>:
  401710:	ff 25 0a 79 20 00    	jmpq   *0x20790a(%rip)        # 609020 <printf@GLIBC_2.2.5>
  401716:	68 01 00 00 00       	pushq  $0x1
  40171b:	e9 d0 ff ff ff       	jmpq   4016f0 <.plt>

0000000000401720 <memcpy@plt>:
  401720:	ff 25 02 79 20 00    	jmpq   *0x207902(%rip)        # 609028 <memcpy@GLIBC_2.14>
  401726:	68 02 00 00 00       	pushq  $0x2
  40172b:	e9 c0 ff ff ff       	jmpq   4016f0 <.plt>

0000000000401730 <_ZN2cv3Mat8copySizeERKS0_@plt>:
  401730:	ff 25 fa 78 20 00    	jmpq   *0x2078fa(%rip)        # 609030 <_ZN2cv3Mat8copySizeERKS0_>
  401736:	68 03 00 00 00       	pushq  $0x3
  40173b:	e9 b0 ff ff ff       	jmpq   4016f0 <.plt>

0000000000401740 <_ZSt17__throw_bad_allocv@plt>:
  401740:	ff 25 f2 78 20 00    	jmpq   *0x2078f2(%rip)        # 609038 <_ZSt17__throw_bad_allocv@GLIBCXX_3.4>
  401746:	68 04 00 00 00       	pushq  $0x4
  40174b:	e9 a0 ff ff ff       	jmpq   4016f0 <.plt>

0000000000401750 <_ZNSt19basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev@plt>:
  401750:	ff 25 ea 78 20 00    	jmpq   *0x2078ea(%rip)        # 609040 <_ZNSt19basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev@GLIBCXX_3.4>
  401756:	68 05 00 00 00       	pushq  $0x5
  40175b:	e9 90 ff ff ff       	jmpq   4016f0 <.plt>

0000000000401760 <_Znwm@plt>:
  401760:	ff 25 e2 78 20 00    	jmpq   *0x2078e2(%rip)        # 609048 <_Znwm@GLIBCXX_3.4>
  401766:	68 06 00 00 00       	pushq  $0x6
  40176b:	e9 80 ff ff ff       	jmpq   4016f0 <.plt>

0000000000401770 <free@plt>:
  401770:	ff 25 da 78 20 00    	jmpq   *0x2078da(%rip)        # 609050 <free@GLIBC_2.2.5>
  401776:	68 07 00 00 00       	pushq  $0x7
  40177b:	e9 70 ff ff ff       	jmpq   4016f0 <.plt>

0000000000401780 <_ZN2cv12_OutputArrayC1ERNS_3MatE@plt>:
  401780:	ff 25 d2 78 20 00    	jmpq   *0x2078d2(%rip)        # 609058 <_ZN2cv12_OutputArrayC1ERNS_3MatE>
  401786:	68 08 00 00 00       	pushq  $0x8
  40178b:	e9 60 ff ff ff       	jmpq   4016f0 <.plt>

0000000000401790 <_ZNSolsEi@plt>:
  401790:	ff 25 ca 78 20 00    	jmpq   *0x2078ca(%rip)        # 609060 <_ZNSolsEi@GLIBCXX_3.4>
  401796:	68 09 00 00 00       	pushq  $0x9
  40179b:	e9 50 ff ff ff       	jmpq   4016f0 <.plt>

00000000004017a0 <_ZNSt19basic_ostringstreamIcSt11char_traitsIcESaIcEE3strERKSs@plt>:
  4017a0:	ff 25 c2 78 20 00    	jmpq   *0x2078c2(%rip)        # 609068 <_ZNSt19basic_ostringstreamIcSt11char_traitsIcESaIcEE3strERKSs@GLIBCXX_3.4>
  4017a6:	68 0a 00 00 00       	pushq  $0xa
  4017ab:	e9 40 ff ff ff       	jmpq   4016f0 <.plt>

00000000004017b0 <__gmon_start__@plt>:
  4017b0:	ff 25 ba 78 20 00    	jmpq   *0x2078ba(%rip)        # 609070 <__gmon_start__>
  4017b6:	68 0b 00 00 00       	pushq  $0xb
  4017bb:	e9 30 ff ff ff       	jmpq   4016f0 <.plt>

00000000004017c0 <_ZN2cv3Mat10deallocateEv@plt>:
  4017c0:	ff 25 b2 78 20 00    	jmpq   *0x2078b2(%rip)        # 609078 <_ZN2cv3Mat10deallocateEv>
  4017c6:	68 0c 00 00 00       	pushq  $0xc
  4017cb:	e9 20 ff ff ff       	jmpq   4016f0 <.plt>

00000000004017d0 <_ZdlPv@plt>:
  4017d0:	ff 25 aa 78 20 00    	jmpq   *0x2078aa(%rip)        # 609080 <_ZdlPv@GLIBCXX_3.4>
  4017d6:	68 0d 00 00 00       	pushq  $0xd
  4017db:	e9 10 ff ff ff       	jmpq   4016f0 <.plt>

00000000004017e0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>:
  4017e0:	ff 25 a2 78 20 00    	jmpq   *0x2078a2(%rip)        # 609088 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@GLIBCXX_3.4>
  4017e6:	68 0e 00 00 00       	pushq  $0xe
  4017eb:	e9 00 ff ff ff       	jmpq   4016f0 <.plt>

00000000004017f0 <_ZNSt19basic_ostringstreamIcSt11char_traitsIcESaIcEEC1ESt13_Ios_Openmode@plt>:
  4017f0:	ff 25 9a 78 20 00    	jmpq   *0x20789a(%rip)        # 609090 <_ZNSt19basic_ostringstreamIcSt11char_traitsIcESaIcEEC1ESt13_Ios_Openmode@GLIBCXX_3.4>
  4017f6:	68 0f 00 00 00       	pushq  $0xf
  4017fb:	e9 f0 fe ff ff       	jmpq   4016f0 <.plt>

0000000000401800 <__gxx_personality_v0@plt>:
  401800:	ff 25 92 78 20 00    	jmpq   *0x207892(%rip)        # 609098 <__gxx_personality_v0@CXXABI_1.3>
  401806:	68 10 00 00 00       	pushq  $0x10
  40180b:	e9 e0 fe ff ff       	jmpq   4016f0 <.plt>

0000000000401810 <_ZSt20__throw_length_errorPKc@plt>:
  401810:	ff 25 8a 78 20 00    	jmpq   *0x20788a(%rip)        # 6090a0 <_ZSt20__throw_length_errorPKc@GLIBCXX_3.4>
  401816:	68 11 00 00 00       	pushq  $0x11
  40181b:	e9 d0 fe ff ff       	jmpq   4016f0 <.plt>

0000000000401820 <_ZNKSt19basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv@plt>:
  401820:	ff 25 82 78 20 00    	jmpq   *0x207882(%rip)        # 6090a8 <_ZNKSt19basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv@GLIBCXX_3.4>
  401826:	68 12 00 00 00       	pushq  $0x12
  40182b:	e9 c0 fe ff ff       	jmpq   4016f0 <.plt>

0000000000401830 <_ZN2cv7imwriteERKSsRKNS_11_InputArrayERKSt6vectorIiSaIiEE@plt>:
  401830:	ff 25 7a 78 20 00    	jmpq   *0x20787a(%rip)        # 6090b0 <_ZN2cv7imwriteERKSsRKNS_11_InputArrayERKSt6vectorIiSaIiEE>
  401836:	68 13 00 00 00       	pushq  $0x13
  40183b:	e9 b0 fe ff ff       	jmpq   4016f0 <.plt>

0000000000401840 <_Unwind_Resume@plt>:
  401840:	ff 25 72 78 20 00    	jmpq   *0x207872(%rip)        # 6090b8 <_Unwind_Resume@GCC_3.0>
  401846:	68 14 00 00 00       	pushq  $0x14
  40184b:	e9 a0 fe ff ff       	jmpq   4016f0 <.plt>

0000000000401850 <_ZNSt8ios_base4InitD1Ev@plt>:
  401850:	ff 25 6a 78 20 00    	jmpq   *0x20786a(%rip)        # 6090c0 <_ZNSt8ios_base4InitD1Ev@GLIBCXX_3.4>
  401856:	68 15 00 00 00       	pushq  $0x15
  40185b:	e9 90 fe ff ff       	jmpq   4016f0 <.plt>

0000000000401860 <_ZN2cv11_InputArrayC1ERKNS_3MatE@plt>:
  401860:	ff 25 62 78 20 00    	jmpq   *0x207862(%rip)        # 6090c8 <_ZN2cv11_InputArrayC1ERKNS_3MatE>
  401866:	68 16 00 00 00       	pushq  $0x16
  40186b:	e9 80 fe ff ff       	jmpq   4016f0 <.plt>

0000000000401870 <_ZN2cv3MatC1ERKS0_RKNS_5RangeES5_@plt>:
  401870:	ff 25 5a 78 20 00    	jmpq   *0x20785a(%rip)        # 6090d0 <_ZN2cv3MatC1ERKS0_RKNS_5RangeES5_>
  401876:	68 17 00 00 00       	pushq  $0x17
  40187b:	e9 70 fe ff ff       	jmpq   4016f0 <.plt>

0000000000401880 <_ZN2cv5mergeERKSt6vectorINS_3MatESaIS1_EERKNS_12_OutputArrayE@plt>:
  401880:	ff 25 52 78 20 00    	jmpq   *0x207852(%rip)        # 6090d8 <_ZN2cv5mergeERKSt6vectorINS_3MatESaIS1_EERKNS_12_OutputArrayE>
  401886:	68 18 00 00 00       	pushq  $0x18
  40188b:	e9 60 fe ff ff       	jmpq   4016f0 <.plt>

0000000000401890 <__cxa_end_catch@plt>:
  401890:	ff 25 4a 78 20 00    	jmpq   *0x20784a(%rip)        # 6090e0 <__cxa_end_catch@CXXABI_1.3>
  401896:	68 19 00 00 00       	pushq  $0x19
  40189b:	e9 50 fe ff ff       	jmpq   4016f0 <.plt>

00000000004018a0 <_ZNSs5clearEv@plt>:
  4018a0:	ff 25 42 78 20 00    	jmpq   *0x207842(%rip)        # 6090e8 <_ZNSs5clearEv@GLIBCXX_3.4>
  4018a6:	68 1a 00 00 00       	pushq  $0x1a
  4018ab:	e9 40 fe ff ff       	jmpq   4016f0 <.plt>

00000000004018b0 <__cxa_rethrow@plt>:
  4018b0:	ff 25 3a 78 20 00    	jmpq   *0x20783a(%rip)        # 6090f0 <__cxa_rethrow@CXXABI_1.3>
  4018b6:	68 1b 00 00 00       	pushq  $0x1b
  4018bb:	e9 30 fe ff ff       	jmpq   4016f0 <.plt>

00000000004018c0 <calloc@plt>:
  4018c0:	ff 25 32 78 20 00    	jmpq   *0x207832(%rip)        # 6090f8 <calloc@GLIBC_2.2.5>
  4018c6:	68 1c 00 00 00       	pushq  $0x1c
  4018cb:	e9 20 fe ff ff       	jmpq   4016f0 <.plt>

00000000004018d0 <__cxa_atexit@plt>:
  4018d0:	ff 25 2a 78 20 00    	jmpq   *0x20782a(%rip)        # 609100 <__cxa_atexit@GLIBC_2.2.5>
  4018d6:	68 1d 00 00 00       	pushq  $0x1d
  4018db:	e9 10 fe ff ff       	jmpq   4016f0 <.plt>

00000000004018e0 <_ZN2cv8fastFreeEPv@plt>:
  4018e0:	ff 25 22 78 20 00    	jmpq   *0x207822(%rip)        # 609108 <_ZN2cv8fastFreeEPv>
  4018e6:	68 1e 00 00 00       	pushq  $0x1e
  4018eb:	e9 00 fe ff ff       	jmpq   4016f0 <.plt>

00000000004018f0 <__pthread_key_create@plt>:
  4018f0:	ff 25 1a 78 20 00    	jmpq   *0x20781a(%rip)        # 609110 <__pthread_key_create>
  4018f6:	68 1f 00 00 00       	pushq  $0x1f
  4018fb:	e9 f0 fd ff ff       	jmpq   4016f0 <.plt>

0000000000401900 <__libc_start_main@plt>:
  401900:	ff 25 12 78 20 00    	jmpq   *0x207812(%rip)        # 609118 <__libc_start_main@GLIBC_2.2.5>
  401906:	68 20 00 00 00       	pushq  $0x20
  40190b:	e9 e0 fd ff ff       	jmpq   4016f0 <.plt>

0000000000401910 <_ZNSaIcEC1Ev@plt>:
  401910:	ff 25 0a 78 20 00    	jmpq   *0x20780a(%rip)        # 609120 <_ZNSaIcEC1Ev@GLIBCXX_3.4>
  401916:	68 21 00 00 00       	pushq  $0x21
  40191b:	e9 d0 fd ff ff       	jmpq   4016f0 <.plt>

0000000000401920 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@plt>:
  401920:	ff 25 02 78 20 00    	jmpq   *0x207802(%rip)        # 609128 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4>
  401926:	68 22 00 00 00       	pushq  $0x22
  40192b:	e9 c0 fd ff ff       	jmpq   4016f0 <.plt>

0000000000401930 <_ZNSt8ios_base4InitC1Ev@plt>:
  401930:	ff 25 fa 77 20 00    	jmpq   *0x2077fa(%rip)        # 609130 <_ZNSt8ios_base4InitC1Ev@GLIBCXX_3.4>
  401936:	68 23 00 00 00       	pushq  $0x23
  40193b:	e9 b0 fd ff ff       	jmpq   4016f0 <.plt>

0000000000401940 <_ZN2cv5splitERKNS_3MatEPS0_@plt>:
  401940:	ff 25 f2 77 20 00    	jmpq   *0x2077f2(%rip)        # 609138 <_ZN2cv5splitERKNS_3MatEPS0_>
  401946:	68 24 00 00 00       	pushq  $0x24
  40194b:	e9 a0 fd ff ff       	jmpq   4016f0 <.plt>

0000000000401950 <_ZNSolsEPFRSoS_E@plt>:
  401950:	ff 25 ea 77 20 00    	jmpq   *0x2077ea(%rip)        # 609140 <_ZNSolsEPFRSoS_E@GLIBCXX_3.4>
  401956:	68 25 00 00 00       	pushq  $0x25
  40195b:	e9 90 fd ff ff       	jmpq   4016f0 <.plt>

0000000000401960 <_ZN2cv6imreadERKSsi@plt>:
  401960:	ff 25 e2 77 20 00    	jmpq   *0x2077e2(%rip)        # 609148 <_ZN2cv6imreadERKSsi>
  401966:	68 26 00 00 00       	pushq  $0x26
  40196b:	e9 80 fd ff ff       	jmpq   4016f0 <.plt>

0000000000401970 <_ZNKSs5c_strEv@plt>:
  401970:	ff 25 da 77 20 00    	jmpq   *0x2077da(%rip)        # 609150 <_ZNKSs5c_strEv@GLIBCXX_3.4>
  401976:	68 27 00 00 00       	pushq  $0x27
  40197b:	e9 70 fd ff ff       	jmpq   4016f0 <.plt>

0000000000401980 <_ZNSsD1Ev@plt>:
  401980:	ff 25 d2 77 20 00    	jmpq   *0x2077d2(%rip)        # 609158 <_ZNSsD1Ev@GLIBCXX_3.4>
  401986:	68 28 00 00 00       	pushq  $0x28
  40198b:	e9 60 fd ff ff       	jmpq   4016f0 <.plt>

0000000000401990 <_ZNSsC1EPKcRKSaIcE@plt>:
  401990:	ff 25 ca 77 20 00    	jmpq   *0x2077ca(%rip)        # 609160 <_ZNSsC1EPKcRKSaIcE@GLIBCXX_3.4>
  401996:	68 29 00 00 00       	pushq  $0x29
  40199b:	e9 50 fd ff ff       	jmpq   4016f0 <.plt>

00000000004019a0 <_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E@plt>:
  4019a0:	ff 25 c2 77 20 00    	jmpq   *0x2077c2(%rip)        # 609168 <_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E@GLIBCXX_3.4>
  4019a6:	68 2a 00 00 00       	pushq  $0x2a
  4019ab:	e9 40 fd ff ff       	jmpq   4016f0 <.plt>

00000000004019b0 <_ZNSaIcED1Ev@plt>:
  4019b0:	ff 25 ba 77 20 00    	jmpq   *0x2077ba(%rip)        # 609170 <_ZNSaIcED1Ev@GLIBCXX_3.4>
  4019b6:	68 2b 00 00 00       	pushq  $0x2b
  4019bb:	e9 30 fd ff ff       	jmpq   4016f0 <.plt>

Disassembly of section .text:

00000000004019c0 <_start>:
  4019c0:	31 ed                	xor    %ebp,%ebp
  4019c2:	49 89 d1             	mov    %rdx,%r9
  4019c5:	5e                   	pop    %rsi
  4019c6:	48 89 e2             	mov    %rsp,%rdx
  4019c9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4019cd:	50                   	push   %rax
  4019ce:	54                   	push   %rsp
  4019cf:	49 c7 c0 f0 6f 40 00 	mov    $0x406ff0,%r8
  4019d6:	48 c7 c1 80 6f 40 00 	mov    $0x406f80,%rcx
  4019dd:	48 c7 c7 f8 45 40 00 	mov    $0x4045f8,%rdi
  4019e4:	e8 17 ff ff ff       	callq  401900 <__libc_start_main@plt>
  4019e9:	f4                   	hlt    
  4019ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004019f0 <deregister_tm_clones>:
  4019f0:	b8 bf 91 60 00       	mov    $0x6091bf,%eax
  4019f5:	55                   	push   %rbp
  4019f6:	48 2d b8 91 60 00    	sub    $0x6091b8,%rax
  4019fc:	48 83 f8 0e          	cmp    $0xe,%rax
  401a00:	48 89 e5             	mov    %rsp,%rbp
  401a03:	77 02                	ja     401a07 <deregister_tm_clones+0x17>
  401a05:	5d                   	pop    %rbp
  401a06:	c3                   	retq   
  401a07:	b8 00 00 00 00       	mov    $0x0,%eax
  401a0c:	48 85 c0             	test   %rax,%rax
  401a0f:	74 f4                	je     401a05 <deregister_tm_clones+0x15>
  401a11:	5d                   	pop    %rbp
  401a12:	bf b8 91 60 00       	mov    $0x6091b8,%edi
  401a17:	ff e0                	jmpq   *%rax
  401a19:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401a20 <register_tm_clones>:
  401a20:	b8 b8 91 60 00       	mov    $0x6091b8,%eax
  401a25:	55                   	push   %rbp
  401a26:	48 2d b8 91 60 00    	sub    $0x6091b8,%rax
  401a2c:	48 c1 f8 03          	sar    $0x3,%rax
  401a30:	48 89 e5             	mov    %rsp,%rbp
  401a33:	48 89 c2             	mov    %rax,%rdx
  401a36:	48 c1 ea 3f          	shr    $0x3f,%rdx
  401a3a:	48 01 d0             	add    %rdx,%rax
  401a3d:	48 d1 f8             	sar    %rax
  401a40:	75 02                	jne    401a44 <register_tm_clones+0x24>
  401a42:	5d                   	pop    %rbp
  401a43:	c3                   	retq   
  401a44:	ba 00 00 00 00       	mov    $0x0,%edx
  401a49:	48 85 d2             	test   %rdx,%rdx
  401a4c:	74 f4                	je     401a42 <register_tm_clones+0x22>
  401a4e:	5d                   	pop    %rbp
  401a4f:	48 89 c6             	mov    %rax,%rsi
  401a52:	bf b8 91 60 00       	mov    $0x6091b8,%edi
  401a57:	ff e2                	jmpq   *%rdx
  401a59:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401a60 <__do_global_dtors_aux>:
  401a60:	80 3d 69 78 20 00 00 	cmpb   $0x0,0x207869(%rip)        # 6092d0 <completed.6355>
  401a67:	75 11                	jne    401a7a <__do_global_dtors_aux+0x1a>
  401a69:	55                   	push   %rbp
  401a6a:	48 89 e5             	mov    %rsp,%rbp
  401a6d:	e8 7e ff ff ff       	callq  4019f0 <deregister_tm_clones>
  401a72:	5d                   	pop    %rbp
  401a73:	c6 05 56 78 20 00 01 	movb   $0x1,0x207856(%rip)        # 6092d0 <completed.6355>
  401a7a:	f3 c3                	repz retq 
  401a7c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401a80 <frame_dummy>:
  401a80:	48 83 3d 68 72 20 00 	cmpq   $0x0,0x207268(%rip)        # 608cf0 <__JCR_END__>
  401a87:	00 
  401a88:	74 1e                	je     401aa8 <frame_dummy+0x28>
  401a8a:	b8 00 00 00 00       	mov    $0x0,%eax
  401a8f:	48 85 c0             	test   %rax,%rax
  401a92:	74 14                	je     401aa8 <frame_dummy+0x28>
  401a94:	55                   	push   %rbp
  401a95:	bf f0 8c 60 00       	mov    $0x608cf0,%edi
  401a9a:	48 89 e5             	mov    %rsp,%rbp
  401a9d:	ff d0                	callq  *%rax
  401a9f:	5d                   	pop    %rbp
  401aa0:	e9 7b ff ff ff       	jmpq   401a20 <register_tm_clones>
  401aa5:	0f 1f 00             	nopl   (%rax)
  401aa8:	e9 73 ff ff ff       	jmpq   401a20 <register_tm_clones>

0000000000401aad <_ZL5rdtscv>:
  401aad:	55                   	push   %rbp
  401aae:	48 89 e5             	mov    %rsp,%rbp
  401ab1:	0f 31                	rdtsc  
  401ab3:	89 45 fc             	mov    %eax,-0x4(%rbp)
  401ab6:	89 55 f8             	mov    %edx,-0x8(%rbp)
  401ab9:	8b 45 fc             	mov    -0x4(%rbp),%eax
  401abc:	8b 55 f8             	mov    -0x8(%rbp),%edx
  401abf:	48 c1 e2 20          	shl    $0x20,%rdx
  401ac3:	48 09 d0             	or     %rdx,%rax
  401ac6:	5d                   	pop    %rbp
  401ac7:	c3                   	retq   

0000000000401ac8 <_Z11decodeImagePfS_S_PKc>:
  401ac8:	55                   	push   %rbp
  401ac9:	48 89 e5             	mov    %rsp,%rbp
  401acc:	41 54                	push   %r12
  401ace:	53                   	push   %rbx
  401acf:	48 81 ec 30 03 00 00 	sub    $0x330,%rsp
  401ad6:	48 89 bd d8 fc ff ff 	mov    %rdi,-0x328(%rbp)
  401add:	48 89 b5 d0 fc ff ff 	mov    %rsi,-0x330(%rbp)
  401ae4:	48 89 95 c8 fc ff ff 	mov    %rdx,-0x338(%rbp)
  401aeb:	48 89 8d c0 fc ff ff 	mov    %rcx,-0x340(%rbp)
  401af2:	c7 45 e4 00 00 00 00 	movl   $0x0,-0x1c(%rbp)
  401af9:	48 8d 85 e0 fc ff ff 	lea    -0x320(%rbp),%rax
  401b00:	48 89 c7             	mov    %rax,%rdi
  401b03:	e8 fa 33 00 00       	callq  404f02 <_ZN2cv3MatC1Ev>
  401b08:	48 8d 85 40 fd ff ff 	lea    -0x2c0(%rbp),%rax
  401b0f:	48 89 c7             	mov    %rax,%rdi
  401b12:	e8 eb 33 00 00       	callq  404f02 <_ZN2cv3MatC1Ev>
  401b17:	48 8d 85 a0 fd ff ff 	lea    -0x260(%rbp),%rax
  401b1e:	bb 02 00 00 00       	mov    $0x2,%ebx
  401b23:	49 89 c4             	mov    %rax,%r12
  401b26:	eb 10                	jmp    401b38 <_Z11decodeImagePfS_S_PKc+0x70>
  401b28:	4c 89 e7             	mov    %r12,%rdi
  401b2b:	e8 d2 33 00 00       	callq  404f02 <_ZN2cv3MatC1Ev>
  401b30:	49 83 c4 60          	add    $0x60,%r12
  401b34:	48 83 eb 01          	sub    $0x1,%rbx
  401b38:	48 83 fb ff          	cmp    $0xffffffffffffffff,%rbx
  401b3c:	75 ea                	jne    401b28 <_Z11decodeImagePfS_S_PKc+0x60>
  401b3e:	48 8d 85 c0 fe ff ff 	lea    -0x140(%rbp),%rax
  401b45:	48 89 c7             	mov    %rax,%rdi
  401b48:	e8 9f 3b 00 00       	callq  4056ec <_ZNSt6vectorIfSaIfEEC1Ev>
  401b4d:	48 8d 85 4f ff ff ff 	lea    -0xb1(%rbp),%rax
  401b54:	48 89 c7             	mov    %rax,%rdi
  401b57:	e8 b4 fd ff ff       	callq  401910 <_ZNSaIcEC1Ev@plt>
  401b5c:	48 8d 95 4f ff ff ff 	lea    -0xb1(%rbp),%rdx
  401b63:	48 8b 8d c0 fc ff ff 	mov    -0x340(%rbp),%rcx
  401b6a:	48 8d 85 40 ff ff ff 	lea    -0xc0(%rbp),%rax
  401b71:	48 89 ce             	mov    %rcx,%rsi
  401b74:	48 89 c7             	mov    %rax,%rdi
  401b77:	e8 14 fe ff ff       	callq  401990 <_ZNSsC1EPKcRKSaIcE@plt>
  401b7c:	48 8d 85 e0 fe ff ff 	lea    -0x120(%rbp),%rax
  401b83:	48 8d 8d 40 ff ff ff 	lea    -0xc0(%rbp),%rcx
  401b8a:	ba 01 00 00 00       	mov    $0x1,%edx
  401b8f:	48 89 ce             	mov    %rcx,%rsi
  401b92:	48 89 c7             	mov    %rax,%rdi
  401b95:	e8 c6 fd ff ff       	callq  401960 <_ZN2cv6imreadERKSsi@plt>
  401b9a:	48 8d 95 e0 fe ff ff 	lea    -0x120(%rbp),%rdx
  401ba1:	48 8d 85 e0 fc ff ff 	lea    -0x320(%rbp),%rax
  401ba8:	48 89 d6             	mov    %rdx,%rsi
  401bab:	48 89 c7             	mov    %rax,%rdi
  401bae:	e8 49 37 00 00       	callq  4052fc <_ZN2cv3MataSERKS0_>
  401bb3:	48 8d 85 e0 fe ff ff 	lea    -0x120(%rbp),%rax
  401bba:	48 89 c7             	mov    %rax,%rdi
  401bbd:	e8 fa 36 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  401bc2:	48 8d 85 40 ff ff ff 	lea    -0xc0(%rbp),%rax
  401bc9:	48 89 c7             	mov    %rax,%rdi
  401bcc:	e8 af fd ff ff       	callq  401980 <_ZNSsD1Ev@plt>
  401bd1:	48 8d 85 4f ff ff ff 	lea    -0xb1(%rbp),%rax
  401bd8:	48 89 c7             	mov    %rax,%rdi
  401bdb:	e8 d0 fd ff ff       	callq  4019b0 <_ZNSaIcED1Ev@plt>
  401be0:	8b 85 e8 fc ff ff    	mov    -0x318(%rbp),%eax
  401be6:	89 45 e0             	mov    %eax,-0x20(%rbp)
  401be9:	8b 85 ec fc ff ff    	mov    -0x314(%rbp),%eax
  401bef:	89 45 dc             	mov    %eax,-0x24(%rbp)
  401bf2:	be f0 70 40 00       	mov    $0x4070f0,%esi
  401bf7:	bf c0 91 60 00       	mov    $0x6091c0,%edi
  401bfc:	e8 df fb ff ff       	callq  4017e0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
  401c01:	8b 55 dc             	mov    -0x24(%rbp),%edx
  401c04:	89 d6                	mov    %edx,%esi
  401c06:	48 89 c7             	mov    %rax,%rdi
  401c09:	e8 82 fb ff ff       	callq  401790 <_ZNSolsEi@plt>
  401c0e:	be 20 19 40 00       	mov    $0x401920,%esi
  401c13:	48 89 c7             	mov    %rax,%rdi
  401c16:	e8 35 fd ff ff       	callq  401950 <_ZNSolsEPFRSoS_E@plt>
  401c1b:	be f9 70 40 00       	mov    $0x4070f9,%esi
  401c20:	bf c0 91 60 00       	mov    $0x6091c0,%edi
  401c25:	e8 b6 fb ff ff       	callq  4017e0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
  401c2a:	8b 55 e0             	mov    -0x20(%rbp),%edx
  401c2d:	89 d6                	mov    %edx,%esi
  401c2f:	48 89 c7             	mov    %rax,%rdi
  401c32:	e8 59 fb ff ff       	callq  401790 <_ZNSolsEi@plt>
  401c37:	be 20 19 40 00       	mov    $0x401920,%esi
  401c3c:	48 89 c7             	mov    %rax,%rdi
  401c3f:	e8 0c fd ff ff       	callq  401950 <_ZNSolsEPFRSoS_E@plt>
  401c44:	c7 45 ec 00 00 00 00 	movl   $0x0,-0x14(%rbp)
  401c4b:	e9 dd 01 00 00       	jmpq   401e2d <_Z11decodeImagePfS_S_PKc+0x365>
  401c50:	c7 45 e8 00 00 00 00 	movl   $0x0,-0x18(%rbp)
  401c57:	e9 be 01 00 00       	jmpq   401e1a <_Z11decodeImagePfS_S_PKc+0x352>
  401c5c:	8b 45 e8             	mov    -0x18(%rbp),%eax
  401c5f:	8d 50 02             	lea    0x2(%rax),%edx
  401c62:	8b 4d e8             	mov    -0x18(%rbp),%ecx
  401c65:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  401c69:	89 ce                	mov    %ecx,%esi
  401c6b:	48 89 c7             	mov    %rax,%rdi
  401c6e:	e8 db 31 00 00       	callq  404e4e <_ZN2cv5RangeC1Eii>
  401c73:	8b 45 ec             	mov    -0x14(%rbp),%eax
  401c76:	8d 50 02             	lea    0x2(%rax),%edx
  401c79:	8b 4d ec             	mov    -0x14(%rbp),%ecx
  401c7c:	48 8d 45 c0          	lea    -0x40(%rbp),%rax
  401c80:	89 ce                	mov    %ecx,%esi
  401c82:	48 89 c7             	mov    %rax,%rdi
  401c85:	e8 c4 31 00 00       	callq  404e4e <_ZN2cv5RangeC1Eii>
  401c8a:	48 8d 85 50 ff ff ff 	lea    -0xb0(%rbp),%rax
  401c91:	48 8b 4d b0          	mov    -0x50(%rbp),%rcx
  401c95:	48 8b 55 c0          	mov    -0x40(%rbp),%rdx
  401c99:	48 8d b5 e0 fc ff ff 	lea    -0x320(%rbp),%rsi
  401ca0:	48 89 c7             	mov    %rax,%rdi
  401ca3:	e8 78 38 00 00       	callq  405520 <_ZNK2cv3MatclENS_5RangeES1_>
  401ca8:	48 8d 95 50 ff ff ff 	lea    -0xb0(%rbp),%rdx
  401caf:	48 8d 85 40 fd ff ff 	lea    -0x2c0(%rbp),%rax
  401cb6:	48 89 d6             	mov    %rdx,%rsi
  401cb9:	48 89 c7             	mov    %rax,%rdi
  401cbc:	e8 3b 36 00 00       	callq  4052fc <_ZN2cv3MataSERKS0_>
  401cc1:	48 8d 85 50 ff ff ff 	lea    -0xb0(%rbp),%rax
  401cc8:	48 89 c7             	mov    %rax,%rdi
  401ccb:	e8 ec 35 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  401cd0:	48 8d 95 a0 fd ff ff 	lea    -0x260(%rbp),%rdx
  401cd7:	48 8d 85 40 fd ff ff 	lea    -0x2c0(%rbp),%rax
  401cde:	48 89 d6             	mov    %rdx,%rsi
  401ce1:	48 89 c7             	mov    %rax,%rdi
  401ce4:	e8 57 fc ff ff       	callq  401940 <_ZN2cv5splitERKNS_3MatEPS0_@plt>
  401ce9:	48 8b 95 c8 fd ff ff 	mov    -0x238(%rbp),%rdx
  401cf0:	48 8b 8d c0 fd ff ff 	mov    -0x240(%rbp),%rcx
  401cf7:	48 8d 85 c0 fe ff ff 	lea    -0x140(%rbp),%rax
  401cfe:	48 89 ce             	mov    %rcx,%rsi
  401d01:	48 89 c7             	mov    %rax,%rdi
  401d04:	e8 63 3a 00 00       	callq  40576c <_ZNSt6vectorIfSaIfEE6assignIPhEEvT_S4_>
  401d09:	48 8d 85 c0 fe ff ff 	lea    -0x140(%rbp),%rax
  401d10:	be 00 00 00 00       	mov    $0x0,%esi
  401d15:	48 89 c7             	mov    %rax,%rdi
  401d18:	e8 81 3a 00 00       	callq  40579e <_ZNSt6vectorIfSaIfEEixEm>
  401d1d:	48 89 45 d0          	mov    %rax,-0x30(%rbp)
  401d21:	ba 10 00 00 00       	mov    $0x10,%edx
  401d26:	8b 45 e4             	mov    -0x1c(%rbp),%eax
  401d29:	48 98                	cltq   
  401d2b:	48 8d 0c 85 00 00 00 	lea    0x0(,%rax,4),%rcx
  401d32:	00 
  401d33:	48 8b 85 c8 fc ff ff 	mov    -0x338(%rbp),%rax
  401d3a:	48 01 c1             	add    %rax,%rcx
  401d3d:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  401d41:	48 89 c6             	mov    %rax,%rsi
  401d44:	48 89 cf             	mov    %rcx,%rdi
  401d47:	e8 d4 f9 ff ff       	callq  401720 <memcpy@plt>
  401d4c:	48 8b 95 28 fe ff ff 	mov    -0x1d8(%rbp),%rdx
  401d53:	48 8b 8d 20 fe ff ff 	mov    -0x1e0(%rbp),%rcx
  401d5a:	48 8d 85 c0 fe ff ff 	lea    -0x140(%rbp),%rax
  401d61:	48 89 ce             	mov    %rcx,%rsi
  401d64:	48 89 c7             	mov    %rax,%rdi
  401d67:	e8 00 3a 00 00       	callq  40576c <_ZNSt6vectorIfSaIfEE6assignIPhEEvT_S4_>
  401d6c:	48 8d 85 c0 fe ff ff 	lea    -0x140(%rbp),%rax
  401d73:	be 00 00 00 00       	mov    $0x0,%esi
  401d78:	48 89 c7             	mov    %rax,%rdi
  401d7b:	e8 1e 3a 00 00       	callq  40579e <_ZNSt6vectorIfSaIfEEixEm>
  401d80:	48 89 45 d0          	mov    %rax,-0x30(%rbp)
  401d84:	ba 10 00 00 00       	mov    $0x10,%edx
  401d89:	8b 45 e4             	mov    -0x1c(%rbp),%eax
  401d8c:	48 98                	cltq   
  401d8e:	48 8d 0c 85 00 00 00 	lea    0x0(,%rax,4),%rcx
  401d95:	00 
  401d96:	48 8b 85 d0 fc ff ff 	mov    -0x330(%rbp),%rax
  401d9d:	48 01 c1             	add    %rax,%rcx
  401da0:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  401da4:	48 89 c6             	mov    %rax,%rsi
  401da7:	48 89 cf             	mov    %rcx,%rdi
  401daa:	e8 71 f9 ff ff       	callq  401720 <memcpy@plt>
  401daf:	48 8b 95 88 fe ff ff 	mov    -0x178(%rbp),%rdx
  401db6:	48 8b 8d 80 fe ff ff 	mov    -0x180(%rbp),%rcx
  401dbd:	48 8d 85 c0 fe ff ff 	lea    -0x140(%rbp),%rax
  401dc4:	48 89 ce             	mov    %rcx,%rsi
  401dc7:	48 89 c7             	mov    %rax,%rdi
  401dca:	e8 9d 39 00 00       	callq  40576c <_ZNSt6vectorIfSaIfEE6assignIPhEEvT_S4_>
  401dcf:	48 8d 85 c0 fe ff ff 	lea    -0x140(%rbp),%rax
  401dd6:	be 00 00 00 00       	mov    $0x0,%esi
  401ddb:	48 89 c7             	mov    %rax,%rdi
  401dde:	e8 bb 39 00 00       	callq  40579e <_ZNSt6vectorIfSaIfEEixEm>
  401de3:	48 89 45 d0          	mov    %rax,-0x30(%rbp)
  401de7:	ba 10 00 00 00       	mov    $0x10,%edx
  401dec:	8b 45 e4             	mov    -0x1c(%rbp),%eax
  401def:	48 98                	cltq   
  401df1:	48 8d 0c 85 00 00 00 	lea    0x0(,%rax,4),%rcx
  401df8:	00 
  401df9:	48 8b 85 d8 fc ff ff 	mov    -0x328(%rbp),%rax
  401e00:	48 01 c1             	add    %rax,%rcx
  401e03:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  401e07:	48 89 c6             	mov    %rax,%rsi
  401e0a:	48 89 cf             	mov    %rcx,%rdi
  401e0d:	e8 0e f9 ff ff       	callq  401720 <memcpy@plt>
  401e12:	83 45 e4 04          	addl   $0x4,-0x1c(%rbp)
  401e16:	83 45 e8 02          	addl   $0x2,-0x18(%rbp)
  401e1a:	8b 45 e8             	mov    -0x18(%rbp),%eax
  401e1d:	83 c0 02             	add    $0x2,%eax
  401e20:	3b 45 dc             	cmp    -0x24(%rbp),%eax
  401e23:	0f 8e 33 fe ff ff    	jle    401c5c <_Z11decodeImagePfS_S_PKc+0x194>
  401e29:	83 45 ec 02          	addl   $0x2,-0x14(%rbp)
  401e2d:	8b 45 ec             	mov    -0x14(%rbp),%eax
  401e30:	83 c0 02             	add    $0x2,%eax
  401e33:	3b 45 e0             	cmp    -0x20(%rbp),%eax
  401e36:	0f 8e 14 fe ff ff    	jle    401c50 <_Z11decodeImagePfS_S_PKc+0x188>
  401e3c:	41 bc 00 00 00 00    	mov    $0x0,%r12d
  401e42:	48 8d 85 c0 fe ff ff 	lea    -0x140(%rbp),%rax
  401e49:	48 89 c7             	mov    %rax,%rdi
  401e4c:	e8 b5 38 00 00       	callq  405706 <_ZNSt6vectorIfSaIfEED1Ev>
  401e51:	48 8d 9d a0 fd ff ff 	lea    -0x260(%rbp),%rbx
  401e58:	48 81 c3 20 01 00 00 	add    $0x120,%rbx
  401e5f:	48 8d 85 a0 fd ff ff 	lea    -0x260(%rbp),%rax
  401e66:	48 39 c3             	cmp    %rax,%rbx
  401e69:	74 0e                	je     401e79 <_Z11decodeImagePfS_S_PKc+0x3b1>
  401e6b:	48 83 eb 60          	sub    $0x60,%rbx
  401e6f:	48 89 df             	mov    %rbx,%rdi
  401e72:	e8 45 34 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  401e77:	eb e6                	jmp    401e5f <_Z11decodeImagePfS_S_PKc+0x397>
  401e79:	48 8d 85 40 fd ff ff 	lea    -0x2c0(%rbp),%rax
  401e80:	48 89 c7             	mov    %rax,%rdi
  401e83:	e8 34 34 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  401e88:	48 8d 85 e0 fc ff ff 	lea    -0x320(%rbp),%rax
  401e8f:	48 89 c7             	mov    %rax,%rdi
  401e92:	e8 25 34 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  401e97:	44 89 e0             	mov    %r12d,%eax
  401e9a:	e9 c8 00 00 00       	jmpq   401f67 <_Z11decodeImagePfS_S_PKc+0x49f>
  401e9f:	48 89 c3             	mov    %rax,%rbx
  401ea2:	48 8d 85 e0 fe ff ff 	lea    -0x120(%rbp),%rax
  401ea9:	48 89 c7             	mov    %rax,%rdi
  401eac:	e8 0b 34 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  401eb1:	eb 03                	jmp    401eb6 <_Z11decodeImagePfS_S_PKc+0x3ee>
  401eb3:	48 89 c3             	mov    %rax,%rbx
  401eb6:	48 8d 85 40 ff ff ff 	lea    -0xc0(%rbp),%rax
  401ebd:	48 89 c7             	mov    %rax,%rdi
  401ec0:	e8 bb fa ff ff       	callq  401980 <_ZNSsD1Ev@plt>
  401ec5:	eb 03                	jmp    401eca <_Z11decodeImagePfS_S_PKc+0x402>
  401ec7:	48 89 c3             	mov    %rax,%rbx
  401eca:	48 8d 85 4f ff ff ff 	lea    -0xb1(%rbp),%rax
  401ed1:	48 89 c7             	mov    %rax,%rdi
  401ed4:	e8 d7 fa ff ff       	callq  4019b0 <_ZNSaIcED1Ev@plt>
  401ed9:	eb 17                	jmp    401ef2 <_Z11decodeImagePfS_S_PKc+0x42a>
  401edb:	48 89 c3             	mov    %rax,%rbx
  401ede:	48 8d 85 50 ff ff ff 	lea    -0xb0(%rbp),%rax
  401ee5:	48 89 c7             	mov    %rax,%rdi
  401ee8:	e8 cf 33 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  401eed:	eb 03                	jmp    401ef2 <_Z11decodeImagePfS_S_PKc+0x42a>
  401eef:	48 89 c3             	mov    %rax,%rbx
  401ef2:	48 8d 85 c0 fe ff ff 	lea    -0x140(%rbp),%rax
  401ef9:	48 89 c7             	mov    %rax,%rdi
  401efc:	e8 05 38 00 00       	callq  405706 <_ZNSt6vectorIfSaIfEED1Ev>
  401f01:	49 89 dc             	mov    %rbx,%r12
  401f04:	eb 03                	jmp    401f09 <_Z11decodeImagePfS_S_PKc+0x441>
  401f06:	49 89 c4             	mov    %rax,%r12
  401f09:	48 8d 9d a0 fd ff ff 	lea    -0x260(%rbp),%rbx
  401f10:	48 81 c3 20 01 00 00 	add    $0x120,%rbx
  401f17:	48 8d 85 a0 fd ff ff 	lea    -0x260(%rbp),%rax
  401f1e:	48 39 c3             	cmp    %rax,%rbx
  401f21:	74 0e                	je     401f31 <_Z11decodeImagePfS_S_PKc+0x469>
  401f23:	48 83 eb 60          	sub    $0x60,%rbx
  401f27:	48 89 df             	mov    %rbx,%rdi
  401f2a:	e8 8d 33 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  401f2f:	eb e6                	jmp    401f17 <_Z11decodeImagePfS_S_PKc+0x44f>
  401f31:	4c 89 e3             	mov    %r12,%rbx
  401f34:	eb 03                	jmp    401f39 <_Z11decodeImagePfS_S_PKc+0x471>
  401f36:	48 89 c3             	mov    %rax,%rbx
  401f39:	48 8d 85 40 fd ff ff 	lea    -0x2c0(%rbp),%rax
  401f40:	48 89 c7             	mov    %rax,%rdi
  401f43:	e8 74 33 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  401f48:	eb 03                	jmp    401f4d <_Z11decodeImagePfS_S_PKc+0x485>
  401f4a:	48 89 c3             	mov    %rax,%rbx
  401f4d:	48 8d 85 e0 fc ff ff 	lea    -0x320(%rbp),%rax
  401f54:	48 89 c7             	mov    %rax,%rdi
  401f57:	e8 60 33 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  401f5c:	48 89 d8             	mov    %rbx,%rax
  401f5f:	48 89 c7             	mov    %rax,%rdi
  401f62:	e8 d9 f8 ff ff       	callq  401840 <_Unwind_Resume@plt>
  401f67:	48 81 c4 30 03 00 00 	add    $0x330,%rsp
  401f6e:	5b                   	pop    %rbx
  401f6f:	41 5c                	pop    %r12
  401f71:	5d                   	pop    %rbp
  401f72:	c3                   	retq   

0000000000401f73 <_Z11encodeImagePfS_S_PKcii>:
  401f73:	55                   	push   %rbp
  401f74:	48 89 e5             	mov    %rsp,%rbp
  401f77:	53                   	push   %rbx
  401f78:	48 81 ec a8 02 00 00 	sub    $0x2a8,%rsp
  401f7f:	48 89 bd 78 fd ff ff 	mov    %rdi,-0x288(%rbp)
  401f86:	48 89 b5 70 fd ff ff 	mov    %rsi,-0x290(%rbp)
  401f8d:	48 89 95 68 fd ff ff 	mov    %rdx,-0x298(%rbp)
  401f94:	48 89 8d 60 fd ff ff 	mov    %rcx,-0x2a0(%rbp)
  401f9b:	44 89 85 5c fd ff ff 	mov    %r8d,-0x2a4(%rbp)
  401fa2:	44 89 8d 58 fd ff ff 	mov    %r9d,-0x2a8(%rbp)
  401fa9:	48 8d 85 60 ff ff ff 	lea    -0xa0(%rbp),%rax
  401fb0:	48 89 c7             	mov    %rax,%rdi
  401fb3:	e8 b4 36 00 00       	callq  40566c <_ZNSt6vectorIN2cv3MatESaIS1_EEC1Ev>
  401fb8:	48 8d 85 80 fd ff ff 	lea    -0x280(%rbp),%rax
  401fbf:	48 89 c7             	mov    %rax,%rdi
  401fc2:	e8 3b 2f 00 00       	callq  404f02 <_ZN2cv3MatC1Ev>
  401fc7:	48 8d 85 e0 fd ff ff 	lea    -0x220(%rbp),%rax
  401fce:	48 89 c7             	mov    %rax,%rdi
  401fd1:	e8 2c 2f 00 00       	callq  404f02 <_ZN2cv3MatC1Ev>
  401fd6:	48 8b 8d 78 fd ff ff 	mov    -0x288(%rbp),%rcx
  401fdd:	8b 95 5c fd ff ff    	mov    -0x2a4(%rbp),%edx
  401fe3:	8b b5 58 fd ff ff    	mov    -0x2a8(%rbp),%esi
  401fe9:	48 8d 85 40 fe ff ff 	lea    -0x1c0(%rbp),%rax
  401ff0:	41 b9 00 00 00 00    	mov    $0x0,%r9d
  401ff6:	49 89 c8             	mov    %rcx,%r8
  401ff9:	b9 05 00 00 00       	mov    $0x5,%ecx
  401ffe:	48 89 c7             	mov    %rax,%rdi
  402001:	e8 c6 30 00 00       	callq  4050cc <_ZN2cv3MatC1EiiiPvm>
  402006:	48 8b 8d 70 fd ff ff 	mov    -0x290(%rbp),%rcx
  40200d:	8b 95 5c fd ff ff    	mov    -0x2a4(%rbp),%edx
  402013:	8b b5 58 fd ff ff    	mov    -0x2a8(%rbp),%esi
  402019:	48 8d 85 a0 fe ff ff 	lea    -0x160(%rbp),%rax
  402020:	41 b9 00 00 00 00    	mov    $0x0,%r9d
  402026:	49 89 c8             	mov    %rcx,%r8
  402029:	b9 05 00 00 00       	mov    $0x5,%ecx
  40202e:	48 89 c7             	mov    %rax,%rdi
  402031:	e8 96 30 00 00       	callq  4050cc <_ZN2cv3MatC1EiiiPvm>
  402036:	48 8b 8d 68 fd ff ff 	mov    -0x298(%rbp),%rcx
  40203d:	8b 95 5c fd ff ff    	mov    -0x2a4(%rbp),%edx
  402043:	8b b5 58 fd ff ff    	mov    -0x2a8(%rbp),%esi
  402049:	48 8d 85 00 ff ff ff 	lea    -0x100(%rbp),%rax
  402050:	41 b9 00 00 00 00    	mov    $0x0,%r9d
  402056:	49 89 c8             	mov    %rcx,%r8
  402059:	b9 05 00 00 00       	mov    $0x5,%ecx
  40205e:	48 89 c7             	mov    %rax,%rdi
  402061:	e8 66 30 00 00       	callq  4050cc <_ZN2cv3MatC1EiiiPvm>
  402066:	48 8d 95 00 ff ff ff 	lea    -0x100(%rbp),%rdx
  40206d:	48 8d 85 60 ff ff ff 	lea    -0xa0(%rbp),%rax
  402074:	48 89 d6             	mov    %rdx,%rsi
  402077:	48 89 c7             	mov    %rax,%rdi
  40207a:	e8 3f 37 00 00       	callq  4057be <_ZNSt6vectorIN2cv3MatESaIS1_EE9push_backERKS1_>
  40207f:	48 8d 95 a0 fe ff ff 	lea    -0x160(%rbp),%rdx
  402086:	48 8d 85 60 ff ff ff 	lea    -0xa0(%rbp),%rax
  40208d:	48 89 d6             	mov    %rdx,%rsi
  402090:	48 89 c7             	mov    %rax,%rdi
  402093:	e8 26 37 00 00       	callq  4057be <_ZNSt6vectorIN2cv3MatESaIS1_EE9push_backERKS1_>
  402098:	48 8d 95 40 fe ff ff 	lea    -0x1c0(%rbp),%rdx
  40209f:	48 8d 85 60 ff ff ff 	lea    -0xa0(%rbp),%rax
  4020a6:	48 89 d6             	mov    %rdx,%rsi
  4020a9:	48 89 c7             	mov    %rax,%rdi
  4020ac:	e8 0d 37 00 00       	callq  4057be <_ZNSt6vectorIN2cv3MatESaIS1_EE9push_backERKS1_>
  4020b1:	48 8d 95 80 fd ff ff 	lea    -0x280(%rbp),%rdx
  4020b8:	48 8d 45 80          	lea    -0x80(%rbp),%rax
  4020bc:	48 89 d6             	mov    %rdx,%rsi
  4020bf:	48 89 c7             	mov    %rax,%rdi
  4020c2:	e8 b9 f6 ff ff       	callq  401780 <_ZN2cv12_OutputArrayC1ERNS_3MatE@plt>
  4020c7:	48 8d 55 80          	lea    -0x80(%rbp),%rdx
  4020cb:	48 8d 85 60 ff ff ff 	lea    -0xa0(%rbp),%rax
  4020d2:	48 89 d6             	mov    %rdx,%rsi
  4020d5:	48 89 c7             	mov    %rax,%rdi
  4020d8:	e8 a3 f7 ff ff       	callq  401880 <_ZN2cv5mergeERKSt6vectorINS_3MatESaIS1_EERKNS_12_OutputArrayE@plt>
  4020dd:	48 8d 45 a0          	lea    -0x60(%rbp),%rax
  4020e1:	48 89 c7             	mov    %rax,%rdi
  4020e4:	e8 03 35 00 00       	callq  4055ec <_ZNSt6vectorIiSaIiEEC1Ev>
  4020e9:	48 8d 95 80 fd ff ff 	lea    -0x280(%rbp),%rdx
  4020f0:	48 8d 45 c0          	lea    -0x40(%rbp),%rax
  4020f4:	48 89 d6             	mov    %rdx,%rsi
  4020f7:	48 89 c7             	mov    %rax,%rdi
  4020fa:	e8 61 f7 ff ff       	callq  401860 <_ZN2cv11_InputArrayC1ERKNS_3MatE@plt>
  4020ff:	48 8d 45 ef          	lea    -0x11(%rbp),%rax
  402103:	48 89 c7             	mov    %rax,%rdi
  402106:	e8 05 f8 ff ff       	callq  401910 <_ZNSaIcEC1Ev@plt>
  40210b:	48 8d 55 ef          	lea    -0x11(%rbp),%rdx
  40210f:	48 8b 8d 60 fd ff ff 	mov    -0x2a0(%rbp),%rcx
  402116:	48 8d 45 e0          	lea    -0x20(%rbp),%rax
  40211a:	48 89 ce             	mov    %rcx,%rsi
  40211d:	48 89 c7             	mov    %rax,%rdi
  402120:	e8 6b f8 ff ff       	callq  401990 <_ZNSsC1EPKcRKSaIcE@plt>
  402125:	48 8d 55 a0          	lea    -0x60(%rbp),%rdx
  402129:	48 8d 4d c0          	lea    -0x40(%rbp),%rcx
  40212d:	48 8d 45 e0          	lea    -0x20(%rbp),%rax
  402131:	48 89 ce             	mov    %rcx,%rsi
  402134:	48 89 c7             	mov    %rax,%rdi
  402137:	e8 f4 f6 ff ff       	callq  401830 <_ZN2cv7imwriteERKSsRKNS_11_InputArrayERKSt6vectorIiSaIiEE@plt>
  40213c:	48 8d 45 e0          	lea    -0x20(%rbp),%rax
  402140:	48 89 c7             	mov    %rax,%rdi
  402143:	e8 38 f8 ff ff       	callq  401980 <_ZNSsD1Ev@plt>
  402148:	48 8d 45 ef          	lea    -0x11(%rbp),%rax
  40214c:	48 89 c7             	mov    %rax,%rdi
  40214f:	e8 5c f8 ff ff       	callq  4019b0 <_ZNSaIcED1Ev@plt>
  402154:	48 8d 45 a0          	lea    -0x60(%rbp),%rax
  402158:	48 89 c7             	mov    %rax,%rdi
  40215b:	e8 a6 34 00 00       	callq  405606 <_ZNSt6vectorIiSaIiEED1Ev>
  402160:	48 8d 85 00 ff ff ff 	lea    -0x100(%rbp),%rax
  402167:	48 89 c7             	mov    %rax,%rdi
  40216a:	e8 4d 31 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  40216f:	48 8d 85 a0 fe ff ff 	lea    -0x160(%rbp),%rax
  402176:	48 89 c7             	mov    %rax,%rdi
  402179:	e8 3e 31 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  40217e:	48 8d 85 40 fe ff ff 	lea    -0x1c0(%rbp),%rax
  402185:	48 89 c7             	mov    %rax,%rdi
  402188:	e8 2f 31 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  40218d:	48 8d 85 e0 fd ff ff 	lea    -0x220(%rbp),%rax
  402194:	48 89 c7             	mov    %rax,%rdi
  402197:	e8 20 31 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  40219c:	48 8d 85 80 fd ff ff 	lea    -0x280(%rbp),%rax
  4021a3:	48 89 c7             	mov    %rax,%rdi
  4021a6:	e8 11 31 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  4021ab:	48 8d 85 60 ff ff ff 	lea    -0xa0(%rbp),%rax
  4021b2:	48 89 c7             	mov    %rax,%rdi
  4021b5:	e8 cc 34 00 00       	callq  405686 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev>
  4021ba:	e9 b4 00 00 00       	jmpq   402273 <_Z11encodeImagePfS_S_PKcii+0x300>
  4021bf:	48 89 c3             	mov    %rax,%rbx
  4021c2:	48 8d 45 e0          	lea    -0x20(%rbp),%rax
  4021c6:	48 89 c7             	mov    %rax,%rdi
  4021c9:	e8 b2 f7 ff ff       	callq  401980 <_ZNSsD1Ev@plt>
  4021ce:	eb 03                	jmp    4021d3 <_Z11encodeImagePfS_S_PKcii+0x260>
  4021d0:	48 89 c3             	mov    %rax,%rbx
  4021d3:	48 8d 45 ef          	lea    -0x11(%rbp),%rax
  4021d7:	48 89 c7             	mov    %rax,%rdi
  4021da:	e8 d1 f7 ff ff       	callq  4019b0 <_ZNSaIcED1Ev@plt>
  4021df:	eb 03                	jmp    4021e4 <_Z11encodeImagePfS_S_PKcii+0x271>
  4021e1:	48 89 c3             	mov    %rax,%rbx
  4021e4:	48 8d 45 a0          	lea    -0x60(%rbp),%rax
  4021e8:	48 89 c7             	mov    %rax,%rdi
  4021eb:	e8 16 34 00 00       	callq  405606 <_ZNSt6vectorIiSaIiEED1Ev>
  4021f0:	eb 03                	jmp    4021f5 <_Z11encodeImagePfS_S_PKcii+0x282>
  4021f2:	48 89 c3             	mov    %rax,%rbx
  4021f5:	48 8d 85 00 ff ff ff 	lea    -0x100(%rbp),%rax
  4021fc:	48 89 c7             	mov    %rax,%rdi
  4021ff:	e8 b8 30 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  402204:	eb 03                	jmp    402209 <_Z11encodeImagePfS_S_PKcii+0x296>
  402206:	48 89 c3             	mov    %rax,%rbx
  402209:	48 8d 85 a0 fe ff ff 	lea    -0x160(%rbp),%rax
  402210:	48 89 c7             	mov    %rax,%rdi
  402213:	e8 a4 30 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  402218:	eb 03                	jmp    40221d <_Z11encodeImagePfS_S_PKcii+0x2aa>
  40221a:	48 89 c3             	mov    %rax,%rbx
  40221d:	48 8d 85 40 fe ff ff 	lea    -0x1c0(%rbp),%rax
  402224:	48 89 c7             	mov    %rax,%rdi
  402227:	e8 90 30 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  40222c:	eb 03                	jmp    402231 <_Z11encodeImagePfS_S_PKcii+0x2be>
  40222e:	48 89 c3             	mov    %rax,%rbx
  402231:	48 8d 85 e0 fd ff ff 	lea    -0x220(%rbp),%rax
  402238:	48 89 c7             	mov    %rax,%rdi
  40223b:	e8 7c 30 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  402240:	eb 03                	jmp    402245 <_Z11encodeImagePfS_S_PKcii+0x2d2>
  402242:	48 89 c3             	mov    %rax,%rbx
  402245:	48 8d 85 80 fd ff ff 	lea    -0x280(%rbp),%rax
  40224c:	48 89 c7             	mov    %rax,%rdi
  40224f:	e8 68 30 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  402254:	eb 03                	jmp    402259 <_Z11encodeImagePfS_S_PKcii+0x2e6>
  402256:	48 89 c3             	mov    %rax,%rbx
  402259:	48 8d 85 60 ff ff ff 	lea    -0xa0(%rbp),%rax
  402260:	48 89 c7             	mov    %rax,%rdi
  402263:	e8 1e 34 00 00       	callq  405686 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev>
  402268:	48 89 d8             	mov    %rbx,%rax
  40226b:	48 89 c7             	mov    %rax,%rdi
  40226e:	e8 cd f5 ff ff       	callq  401840 <_Unwind_Resume@plt>
  402273:	48 81 c4 a8 02 00 00 	add    $0x2a8,%rsp
  40227a:	5b                   	pop    %rbx
  40227b:	5d                   	pop    %rbp
  40227c:	c3                   	retq   

000000000040227d <_Z20generateCoefficientsPf>:
  40227d:	55                   	push   %rbp
  40227e:	48 89 e5             	mov    %rsp,%rbp
  402281:	48 81 ec 50 03 00 00 	sub    $0x350,%rsp
  402288:	48 89 bd 38 fc ff ff 	mov    %rdi,-0x3c8(%rbp)
  40228f:	8b 05 d3 4e 00 00    	mov    0x4ed3(%rip),%eax        # 407168 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x8>
  402295:	89 85 30 fd ff ff    	mov    %eax,-0x2d0(%rbp)
  40229b:	8b 05 cb 4e 00 00    	mov    0x4ecb(%rip),%eax        # 40716c <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0xc>
  4022a1:	89 85 34 fd ff ff    	mov    %eax,-0x2cc(%rbp)
  4022a7:	8b 05 c3 4e 00 00    	mov    0x4ec3(%rip),%eax        # 407170 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x10>
  4022ad:	89 85 38 fd ff ff    	mov    %eax,-0x2c8(%rbp)
  4022b3:	8b 05 bb 4e 00 00    	mov    0x4ebb(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  4022b9:	89 85 3c fd ff ff    	mov    %eax,-0x2c4(%rbp)
  4022bf:	8b 05 af 4e 00 00    	mov    0x4eaf(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  4022c5:	89 85 20 fd ff ff    	mov    %eax,-0x2e0(%rbp)
  4022cb:	8b 05 9f 4e 00 00    	mov    0x4e9f(%rip),%eax        # 407170 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x10>
  4022d1:	89 85 24 fd ff ff    	mov    %eax,-0x2dc(%rbp)
  4022d7:	8b 05 9b 4e 00 00    	mov    0x4e9b(%rip),%eax        # 407178 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x18>
  4022dd:	89 85 28 fd ff ff    	mov    %eax,-0x2d8(%rbp)
  4022e3:	8b 05 7f 4e 00 00    	mov    0x4e7f(%rip),%eax        # 407168 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x8>
  4022e9:	89 85 2c fd ff ff    	mov    %eax,-0x2d4(%rbp)
  4022ef:	8b 05 7f 4e 00 00    	mov    0x4e7f(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  4022f5:	89 85 10 fd ff ff    	mov    %eax,-0x2f0(%rbp)
  4022fb:	8b 05 73 4e 00 00    	mov    0x4e73(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  402301:	89 85 14 fd ff ff    	mov    %eax,-0x2ec(%rbp)
  402307:	8b 05 67 4e 00 00    	mov    0x4e67(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  40230d:	89 85 18 fd ff ff    	mov    %eax,-0x2e8(%rbp)
  402313:	8b 05 5b 4e 00 00    	mov    0x4e5b(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  402319:	89 85 1c fd ff ff    	mov    %eax,-0x2e4(%rbp)
  40231f:	8b 05 4f 4e 00 00    	mov    0x4e4f(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  402325:	89 85 00 fd ff ff    	mov    %eax,-0x300(%rbp)
  40232b:	8b 05 43 4e 00 00    	mov    0x4e43(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  402331:	89 85 04 fd ff ff    	mov    %eax,-0x2fc(%rbp)
  402337:	8b 05 37 4e 00 00    	mov    0x4e37(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  40233d:	89 85 08 fd ff ff    	mov    %eax,-0x2f8(%rbp)
  402343:	8b 05 2b 4e 00 00    	mov    0x4e2b(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  402349:	89 85 0c fd ff ff    	mov    %eax,-0x2f4(%rbp)
  40234f:	48 8d 85 30 fd ff ff 	lea    -0x2d0(%rbp),%rax
  402356:	48 89 45 b8          	mov    %rax,-0x48(%rbp)
  40235a:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
  40235e:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  402362:	c5 f8 29 45 f0       	vmovaps %xmm0,-0x10(%rbp)
  402367:	48 8d 85 20 fd ff ff 	lea    -0x2e0(%rbp),%rax
  40236e:	48 89 45 b0          	mov    %rax,-0x50(%rbp)
  402372:	48 8b 45 b0          	mov    -0x50(%rbp),%rax
  402376:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  40237a:	c5 f8 29 45 e0       	vmovaps %xmm0,-0x20(%rbp)
  40237f:	48 8d 85 10 fd ff ff 	lea    -0x2f0(%rbp),%rax
  402386:	48 89 45 a8          	mov    %rax,-0x58(%rbp)
  40238a:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
  40238e:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  402392:	c5 f8 29 45 d0       	vmovaps %xmm0,-0x30(%rbp)
  402397:	48 8d 85 00 fd ff ff 	lea    -0x300(%rbp),%rax
  40239e:	48 89 45 a0          	mov    %rax,-0x60(%rbp)
  4023a2:	48 8b 45 a0          	mov    -0x60(%rbp),%rax
  4023a6:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  4023aa:	c5 f8 29 45 c0       	vmovaps %xmm0,-0x40(%rbp)
  4023af:	48 8b 85 38 fc ff ff 	mov    -0x3c8(%rbp),%rax
  4023b6:	48 89 45 98          	mov    %rax,-0x68(%rbp)
  4023ba:	c5 f8 28 45 f0       	vmovaps -0x10(%rbp),%xmm0
  4023bf:	c5 f8 29 45 80       	vmovaps %xmm0,-0x80(%rbp)
  4023c4:	48 8b 45 98          	mov    -0x68(%rbp),%rax
  4023c8:	c5 f8 28 45 80       	vmovaps -0x80(%rbp),%xmm0
  4023cd:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  4023d1:	48 8b 85 38 fc ff ff 	mov    -0x3c8(%rbp),%rax
  4023d8:	48 83 c0 10          	add    $0x10,%rax
  4023dc:	48 89 85 78 ff ff ff 	mov    %rax,-0x88(%rbp)
  4023e3:	c5 f8 28 45 e0       	vmovaps -0x20(%rbp),%xmm0
  4023e8:	c5 f8 29 85 60 ff ff 	vmovaps %xmm0,-0xa0(%rbp)
  4023ef:	ff 
  4023f0:	48 8b 85 78 ff ff ff 	mov    -0x88(%rbp),%rax
  4023f7:	c5 f8 28 85 60 ff ff 	vmovaps -0xa0(%rbp),%xmm0
  4023fe:	ff 
  4023ff:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  402403:	48 8b 85 38 fc ff ff 	mov    -0x3c8(%rbp),%rax
  40240a:	48 83 c0 20          	add    $0x20,%rax
  40240e:	48 89 85 58 ff ff ff 	mov    %rax,-0xa8(%rbp)
  402415:	c5 f8 28 45 d0       	vmovaps -0x30(%rbp),%xmm0
  40241a:	c5 f8 29 85 40 ff ff 	vmovaps %xmm0,-0xc0(%rbp)
  402421:	ff 
  402422:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  402429:	c5 f8 28 85 40 ff ff 	vmovaps -0xc0(%rbp),%xmm0
  402430:	ff 
  402431:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  402435:	48 8b 85 38 fc ff ff 	mov    -0x3c8(%rbp),%rax
  40243c:	48 83 c0 30          	add    $0x30,%rax
  402440:	48 89 85 38 ff ff ff 	mov    %rax,-0xc8(%rbp)
  402447:	c5 f8 28 45 c0       	vmovaps -0x40(%rbp),%xmm0
  40244c:	c5 f8 29 85 20 ff ff 	vmovaps %xmm0,-0xe0(%rbp)
  402453:	ff 
  402454:	48 8b 85 38 ff ff ff 	mov    -0xc8(%rbp),%rax
  40245b:	c5 f8 28 85 20 ff ff 	vmovaps -0xe0(%rbp),%xmm0
  402462:	ff 
  402463:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  402467:	8b 05 ff 4c 00 00    	mov    0x4cff(%rip),%eax        # 40716c <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0xc>
  40246d:	89 85 f0 fc ff ff    	mov    %eax,-0x310(%rbp)
  402473:	8b 05 03 4d 00 00    	mov    0x4d03(%rip),%eax        # 40717c <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x1c>
  402479:	89 85 f4 fc ff ff    	mov    %eax,-0x30c(%rbp)
  40247f:	8b 05 fb 4c 00 00    	mov    0x4cfb(%rip),%eax        # 407180 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x20>
  402485:	89 85 f8 fc ff ff    	mov    %eax,-0x308(%rbp)
  40248b:	8b 05 e3 4c 00 00    	mov    0x4ce3(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  402491:	89 85 fc fc ff ff    	mov    %eax,-0x304(%rbp)
  402497:	8b 05 d7 4c 00 00    	mov    0x4cd7(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  40249d:	89 85 e0 fc ff ff    	mov    %eax,-0x320(%rbp)
  4024a3:	8b 05 d7 4c 00 00    	mov    0x4cd7(%rip),%eax        # 407180 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x20>
  4024a9:	89 85 e4 fc ff ff    	mov    %eax,-0x31c(%rbp)
  4024af:	8b 05 c7 4c 00 00    	mov    0x4cc7(%rip),%eax        # 40717c <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x1c>
  4024b5:	89 85 e8 fc ff ff    	mov    %eax,-0x318(%rbp)
  4024bb:	8b 05 ab 4c 00 00    	mov    0x4cab(%rip),%eax        # 40716c <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0xc>
  4024c1:	89 85 ec fc ff ff    	mov    %eax,-0x314(%rbp)
  4024c7:	8b 05 a3 4c 00 00    	mov    0x4ca3(%rip),%eax        # 407170 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x10>
  4024cd:	89 85 d0 fc ff ff    	mov    %eax,-0x330(%rbp)
  4024d3:	8b 05 a7 4c 00 00    	mov    0x4ca7(%rip),%eax        # 407180 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x20>
  4024d9:	89 85 d4 fc ff ff    	mov    %eax,-0x32c(%rbp)
  4024df:	8b 05 9f 4c 00 00    	mov    0x4c9f(%rip),%eax        # 407184 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x24>
  4024e5:	89 85 d8 fc ff ff    	mov    %eax,-0x328(%rbp)
  4024eb:	8b 05 83 4c 00 00    	mov    0x4c83(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  4024f1:	89 85 dc fc ff ff    	mov    %eax,-0x324(%rbp)
  4024f7:	8b 05 77 4c 00 00    	mov    0x4c77(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  4024fd:	89 85 c0 fc ff ff    	mov    %eax,-0x340(%rbp)
  402503:	8b 05 7b 4c 00 00    	mov    0x4c7b(%rip),%eax        # 407184 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x24>
  402509:	89 85 c4 fc ff ff    	mov    %eax,-0x33c(%rbp)
  40250f:	8b 05 6b 4c 00 00    	mov    0x4c6b(%rip),%eax        # 407180 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x20>
  402515:	89 85 c8 fc ff ff    	mov    %eax,-0x338(%rbp)
  40251b:	8b 05 4f 4c 00 00    	mov    0x4c4f(%rip),%eax        # 407170 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x10>
  402521:	89 85 cc fc ff ff    	mov    %eax,-0x334(%rbp)
  402527:	48 8d 85 f0 fc ff ff 	lea    -0x310(%rbp),%rax
  40252e:	48 89 85 18 ff ff ff 	mov    %rax,-0xe8(%rbp)
  402535:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
  40253c:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  402540:	c5 f8 29 45 f0       	vmovaps %xmm0,-0x10(%rbp)
  402545:	48 8d 85 e0 fc ff ff 	lea    -0x320(%rbp),%rax
  40254c:	48 89 85 10 ff ff ff 	mov    %rax,-0xf0(%rbp)
  402553:	48 8b 85 10 ff ff ff 	mov    -0xf0(%rbp),%rax
  40255a:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  40255e:	c5 f8 29 45 e0       	vmovaps %xmm0,-0x20(%rbp)
  402563:	48 8d 85 d0 fc ff ff 	lea    -0x330(%rbp),%rax
  40256a:	48 89 85 08 ff ff ff 	mov    %rax,-0xf8(%rbp)
  402571:	48 8b 85 08 ff ff ff 	mov    -0xf8(%rbp),%rax
  402578:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  40257c:	c5 f8 29 45 d0       	vmovaps %xmm0,-0x30(%rbp)
  402581:	48 8d 85 c0 fc ff ff 	lea    -0x340(%rbp),%rax
  402588:	48 89 85 00 ff ff ff 	mov    %rax,-0x100(%rbp)
  40258f:	48 8b 85 00 ff ff ff 	mov    -0x100(%rbp),%rax
  402596:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  40259a:	c5 f8 29 45 c0       	vmovaps %xmm0,-0x40(%rbp)
  40259f:	48 8b 85 38 fc ff ff 	mov    -0x3c8(%rbp),%rax
  4025a6:	48 83 c0 40          	add    $0x40,%rax
  4025aa:	48 89 85 f8 fe ff ff 	mov    %rax,-0x108(%rbp)
  4025b1:	c5 f8 28 45 f0       	vmovaps -0x10(%rbp),%xmm0
  4025b6:	c5 f8 29 85 e0 fe ff 	vmovaps %xmm0,-0x120(%rbp)
  4025bd:	ff 
  4025be:	48 8b 85 f8 fe ff ff 	mov    -0x108(%rbp),%rax
  4025c5:	c5 f8 28 85 e0 fe ff 	vmovaps -0x120(%rbp),%xmm0
  4025cc:	ff 
  4025cd:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  4025d1:	48 8b 85 38 fc ff ff 	mov    -0x3c8(%rbp),%rax
  4025d8:	48 83 c0 50          	add    $0x50,%rax
  4025dc:	48 89 85 d8 fe ff ff 	mov    %rax,-0x128(%rbp)
  4025e3:	c5 f8 28 45 e0       	vmovaps -0x20(%rbp),%xmm0
  4025e8:	c5 f8 29 85 c0 fe ff 	vmovaps %xmm0,-0x140(%rbp)
  4025ef:	ff 
  4025f0:	48 8b 85 d8 fe ff ff 	mov    -0x128(%rbp),%rax
  4025f7:	c5 f8 28 85 c0 fe ff 	vmovaps -0x140(%rbp),%xmm0
  4025fe:	ff 
  4025ff:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  402603:	48 8b 85 38 fc ff ff 	mov    -0x3c8(%rbp),%rax
  40260a:	48 83 c0 60          	add    $0x60,%rax
  40260e:	48 89 85 b8 fe ff ff 	mov    %rax,-0x148(%rbp)
  402615:	c5 f8 28 45 d0       	vmovaps -0x30(%rbp),%xmm0
  40261a:	c5 f8 29 85 a0 fe ff 	vmovaps %xmm0,-0x160(%rbp)
  402621:	ff 
  402622:	48 8b 85 b8 fe ff ff 	mov    -0x148(%rbp),%rax
  402629:	c5 f8 28 85 a0 fe ff 	vmovaps -0x160(%rbp),%xmm0
  402630:	ff 
  402631:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  402635:	48 8b 85 38 fc ff ff 	mov    -0x3c8(%rbp),%rax
  40263c:	48 83 c0 70          	add    $0x70,%rax
  402640:	48 89 85 98 fe ff ff 	mov    %rax,-0x168(%rbp)
  402647:	c5 f8 28 45 c0       	vmovaps -0x40(%rbp),%xmm0
  40264c:	c5 f8 29 85 80 fe ff 	vmovaps %xmm0,-0x180(%rbp)
  402653:	ff 
  402654:	48 8b 85 98 fe ff ff 	mov    -0x168(%rbp),%rax
  40265b:	c5 f8 28 85 80 fe ff 	vmovaps -0x180(%rbp),%xmm0
  402662:	ff 
  402663:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  402667:	8b 05 03 4b 00 00    	mov    0x4b03(%rip),%eax        # 407170 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x10>
  40266d:	89 85 b0 fc ff ff    	mov    %eax,-0x350(%rbp)
  402673:	8b 05 07 4b 00 00    	mov    0x4b07(%rip),%eax        # 407180 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x20>
  402679:	89 85 b4 fc ff ff    	mov    %eax,-0x34c(%rbp)
  40267f:	8b 05 ff 4a 00 00    	mov    0x4aff(%rip),%eax        # 407184 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x24>
  402685:	89 85 b8 fc ff ff    	mov    %eax,-0x348(%rbp)
  40268b:	8b 05 e3 4a 00 00    	mov    0x4ae3(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  402691:	89 85 bc fc ff ff    	mov    %eax,-0x344(%rbp)
  402697:	8b 05 d7 4a 00 00    	mov    0x4ad7(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  40269d:	89 85 a0 fc ff ff    	mov    %eax,-0x360(%rbp)
  4026a3:	8b 05 db 4a 00 00    	mov    0x4adb(%rip),%eax        # 407184 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x24>
  4026a9:	89 85 a4 fc ff ff    	mov    %eax,-0x35c(%rbp)
  4026af:	8b 05 cb 4a 00 00    	mov    0x4acb(%rip),%eax        # 407180 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x20>
  4026b5:	89 85 a8 fc ff ff    	mov    %eax,-0x358(%rbp)
  4026bb:	8b 05 af 4a 00 00    	mov    0x4aaf(%rip),%eax        # 407170 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x10>
  4026c1:	89 85 ac fc ff ff    	mov    %eax,-0x354(%rbp)
  4026c7:	8b 05 9f 4a 00 00    	mov    0x4a9f(%rip),%eax        # 40716c <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0xc>
  4026cd:	89 85 90 fc ff ff    	mov    %eax,-0x370(%rbp)
  4026d3:	8b 05 a3 4a 00 00    	mov    0x4aa3(%rip),%eax        # 40717c <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x1c>
  4026d9:	89 85 94 fc ff ff    	mov    %eax,-0x36c(%rbp)
  4026df:	8b 05 a3 4a 00 00    	mov    0x4aa3(%rip),%eax        # 407188 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x28>
  4026e5:	89 85 98 fc ff ff    	mov    %eax,-0x368(%rbp)
  4026eb:	8b 05 83 4a 00 00    	mov    0x4a83(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  4026f1:	89 85 9c fc ff ff    	mov    %eax,-0x364(%rbp)
  4026f7:	8b 05 77 4a 00 00    	mov    0x4a77(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  4026fd:	89 85 80 fc ff ff    	mov    %eax,-0x380(%rbp)
  402703:	8b 05 7f 4a 00 00    	mov    0x4a7f(%rip),%eax        # 407188 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x28>
  402709:	89 85 84 fc ff ff    	mov    %eax,-0x37c(%rbp)
  40270f:	8b 05 67 4a 00 00    	mov    0x4a67(%rip),%eax        # 40717c <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x1c>
  402715:	89 85 88 fc ff ff    	mov    %eax,-0x378(%rbp)
  40271b:	8b 05 4b 4a 00 00    	mov    0x4a4b(%rip),%eax        # 40716c <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0xc>
  402721:	89 85 8c fc ff ff    	mov    %eax,-0x374(%rbp)
  402727:	48 8d 85 b0 fc ff ff 	lea    -0x350(%rbp),%rax
  40272e:	48 89 85 78 fe ff ff 	mov    %rax,-0x188(%rbp)
  402735:	48 8b 85 78 fe ff ff 	mov    -0x188(%rbp),%rax
  40273c:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  402740:	c5 f8 29 45 f0       	vmovaps %xmm0,-0x10(%rbp)
  402745:	48 8d 85 a0 fc ff ff 	lea    -0x360(%rbp),%rax
  40274c:	48 89 85 70 fe ff ff 	mov    %rax,-0x190(%rbp)
  402753:	48 8b 85 70 fe ff ff 	mov    -0x190(%rbp),%rax
  40275a:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  40275e:	c5 f8 29 45 e0       	vmovaps %xmm0,-0x20(%rbp)
  402763:	48 8d 85 90 fc ff ff 	lea    -0x370(%rbp),%rax
  40276a:	48 89 85 68 fe ff ff 	mov    %rax,-0x198(%rbp)
  402771:	48 8b 85 68 fe ff ff 	mov    -0x198(%rbp),%rax
  402778:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  40277c:	c5 f8 29 45 d0       	vmovaps %xmm0,-0x30(%rbp)
  402781:	48 8d 85 80 fc ff ff 	lea    -0x380(%rbp),%rax
  402788:	48 89 85 60 fe ff ff 	mov    %rax,-0x1a0(%rbp)
  40278f:	48 8b 85 60 fe ff ff 	mov    -0x1a0(%rbp),%rax
  402796:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  40279a:	c5 f8 29 45 c0       	vmovaps %xmm0,-0x40(%rbp)
  40279f:	48 8b 85 38 fc ff ff 	mov    -0x3c8(%rbp),%rax
  4027a6:	48 83 e8 80          	sub    $0xffffffffffffff80,%rax
  4027aa:	48 89 85 58 fe ff ff 	mov    %rax,-0x1a8(%rbp)
  4027b1:	c5 f8 28 45 f0       	vmovaps -0x10(%rbp),%xmm0
  4027b6:	c5 f8 29 85 40 fe ff 	vmovaps %xmm0,-0x1c0(%rbp)
  4027bd:	ff 
  4027be:	48 8b 85 58 fe ff ff 	mov    -0x1a8(%rbp),%rax
  4027c5:	c5 f8 28 85 40 fe ff 	vmovaps -0x1c0(%rbp),%xmm0
  4027cc:	ff 
  4027cd:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  4027d1:	48 8b 85 38 fc ff ff 	mov    -0x3c8(%rbp),%rax
  4027d8:	48 05 90 00 00 00    	add    $0x90,%rax
  4027de:	48 89 85 38 fe ff ff 	mov    %rax,-0x1c8(%rbp)
  4027e5:	c5 f8 28 45 e0       	vmovaps -0x20(%rbp),%xmm0
  4027ea:	c5 f8 29 85 20 fe ff 	vmovaps %xmm0,-0x1e0(%rbp)
  4027f1:	ff 
  4027f2:	48 8b 85 38 fe ff ff 	mov    -0x1c8(%rbp),%rax
  4027f9:	c5 f8 28 85 20 fe ff 	vmovaps -0x1e0(%rbp),%xmm0
  402800:	ff 
  402801:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  402805:	48 8b 85 38 fc ff ff 	mov    -0x3c8(%rbp),%rax
  40280c:	48 05 a0 00 00 00    	add    $0xa0,%rax
  402812:	48 89 85 18 fe ff ff 	mov    %rax,-0x1e8(%rbp)
  402819:	c5 f8 28 45 d0       	vmovaps -0x30(%rbp),%xmm0
  40281e:	c5 f8 29 85 00 fe ff 	vmovaps %xmm0,-0x200(%rbp)
  402825:	ff 
  402826:	48 8b 85 18 fe ff ff 	mov    -0x1e8(%rbp),%rax
  40282d:	c5 f8 28 85 00 fe ff 	vmovaps -0x200(%rbp),%xmm0
  402834:	ff 
  402835:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  402839:	48 8b 85 38 fc ff ff 	mov    -0x3c8(%rbp),%rax
  402840:	48 05 b0 00 00 00    	add    $0xb0,%rax
  402846:	48 89 85 f8 fd ff ff 	mov    %rax,-0x208(%rbp)
  40284d:	c5 f8 28 45 c0       	vmovaps -0x40(%rbp),%xmm0
  402852:	c5 f8 29 85 e0 fd ff 	vmovaps %xmm0,-0x220(%rbp)
  402859:	ff 
  40285a:	48 8b 85 f8 fd ff ff 	mov    -0x208(%rbp),%rax
  402861:	c5 f8 28 85 e0 fd ff 	vmovaps -0x220(%rbp),%xmm0
  402868:	ff 
  402869:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  40286d:	8b 05 01 49 00 00    	mov    0x4901(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  402873:	89 85 70 fc ff ff    	mov    %eax,-0x390(%rbp)
  402879:	8b 05 f5 48 00 00    	mov    0x48f5(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  40287f:	89 85 74 fc ff ff    	mov    %eax,-0x38c(%rbp)
  402885:	8b 05 e9 48 00 00    	mov    0x48e9(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  40288b:	89 85 78 fc ff ff    	mov    %eax,-0x388(%rbp)
  402891:	8b 05 dd 48 00 00    	mov    0x48dd(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  402897:	89 85 7c fc ff ff    	mov    %eax,-0x384(%rbp)
  40289d:	8b 05 d1 48 00 00    	mov    0x48d1(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  4028a3:	89 85 60 fc ff ff    	mov    %eax,-0x3a0(%rbp)
  4028a9:	8b 05 c5 48 00 00    	mov    0x48c5(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  4028af:	89 85 64 fc ff ff    	mov    %eax,-0x39c(%rbp)
  4028b5:	8b 05 b9 48 00 00    	mov    0x48b9(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  4028bb:	89 85 68 fc ff ff    	mov    %eax,-0x398(%rbp)
  4028c1:	8b 05 ad 48 00 00    	mov    0x48ad(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  4028c7:	89 85 6c fc ff ff    	mov    %eax,-0x394(%rbp)
  4028cd:	8b 05 95 48 00 00    	mov    0x4895(%rip),%eax        # 407168 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x8>
  4028d3:	89 85 50 fc ff ff    	mov    %eax,-0x3b0(%rbp)
  4028d9:	8b 05 8d 48 00 00    	mov    0x488d(%rip),%eax        # 40716c <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0xc>
  4028df:	89 85 54 fc ff ff    	mov    %eax,-0x3ac(%rbp)
  4028e5:	8b 05 85 48 00 00    	mov    0x4885(%rip),%eax        # 407170 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x10>
  4028eb:	89 85 58 fc ff ff    	mov    %eax,-0x3a8(%rbp)
  4028f1:	8b 05 7d 48 00 00    	mov    0x487d(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  4028f7:	89 85 5c fc ff ff    	mov    %eax,-0x3a4(%rbp)
  4028fd:	8b 05 71 48 00 00    	mov    0x4871(%rip),%eax        # 407174 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x14>
  402903:	89 85 40 fc ff ff    	mov    %eax,-0x3c0(%rbp)
  402909:	8b 05 61 48 00 00    	mov    0x4861(%rip),%eax        # 407170 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x10>
  40290f:	89 85 44 fc ff ff    	mov    %eax,-0x3bc(%rbp)
  402915:	8b 05 51 48 00 00    	mov    0x4851(%rip),%eax        # 40716c <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0xc>
  40291b:	89 85 48 fc ff ff    	mov    %eax,-0x3b8(%rbp)
  402921:	8b 05 41 48 00 00    	mov    0x4841(%rip),%eax        # 407168 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x8>
  402927:	89 85 4c fc ff ff    	mov    %eax,-0x3b4(%rbp)
  40292d:	48 8d 85 70 fc ff ff 	lea    -0x390(%rbp),%rax
  402934:	48 89 85 d8 fd ff ff 	mov    %rax,-0x228(%rbp)
  40293b:	48 8b 85 d8 fd ff ff 	mov    -0x228(%rbp),%rax
  402942:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  402946:	c5 f8 29 45 f0       	vmovaps %xmm0,-0x10(%rbp)
  40294b:	48 8d 85 60 fc ff ff 	lea    -0x3a0(%rbp),%rax
  402952:	48 89 85 d0 fd ff ff 	mov    %rax,-0x230(%rbp)
  402959:	48 8b 85 d0 fd ff ff 	mov    -0x230(%rbp),%rax
  402960:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  402964:	c5 f8 29 45 e0       	vmovaps %xmm0,-0x20(%rbp)
  402969:	48 8d 85 50 fc ff ff 	lea    -0x3b0(%rbp),%rax
  402970:	48 89 85 c8 fd ff ff 	mov    %rax,-0x238(%rbp)
  402977:	48 8b 85 c8 fd ff ff 	mov    -0x238(%rbp),%rax
  40297e:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  402982:	c5 f8 29 45 d0       	vmovaps %xmm0,-0x30(%rbp)
  402987:	48 8d 85 40 fc ff ff 	lea    -0x3c0(%rbp),%rax
  40298e:	48 89 85 c0 fd ff ff 	mov    %rax,-0x240(%rbp)
  402995:	48 8b 85 c0 fd ff ff 	mov    -0x240(%rbp),%rax
  40299c:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  4029a0:	c5 f8 29 45 c0       	vmovaps %xmm0,-0x40(%rbp)
  4029a5:	48 8b 85 38 fc ff ff 	mov    -0x3c8(%rbp),%rax
  4029ac:	48 05 c0 00 00 00    	add    $0xc0,%rax
  4029b2:	48 89 85 b8 fd ff ff 	mov    %rax,-0x248(%rbp)
  4029b9:	c5 f8 28 45 f0       	vmovaps -0x10(%rbp),%xmm0
  4029be:	c5 f8 29 85 a0 fd ff 	vmovaps %xmm0,-0x260(%rbp)
  4029c5:	ff 
  4029c6:	48 8b 85 b8 fd ff ff 	mov    -0x248(%rbp),%rax
  4029cd:	c5 f8 28 85 a0 fd ff 	vmovaps -0x260(%rbp),%xmm0
  4029d4:	ff 
  4029d5:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  4029d9:	48 8b 85 38 fc ff ff 	mov    -0x3c8(%rbp),%rax
  4029e0:	48 05 d0 00 00 00    	add    $0xd0,%rax
  4029e6:	48 89 85 98 fd ff ff 	mov    %rax,-0x268(%rbp)
  4029ed:	c5 f8 28 45 e0       	vmovaps -0x20(%rbp),%xmm0
  4029f2:	c5 f8 29 85 80 fd ff 	vmovaps %xmm0,-0x280(%rbp)
  4029f9:	ff 
  4029fa:	48 8b 85 98 fd ff ff 	mov    -0x268(%rbp),%rax
  402a01:	c5 f8 28 85 80 fd ff 	vmovaps -0x280(%rbp),%xmm0
  402a08:	ff 
  402a09:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  402a0d:	48 8b 85 38 fc ff ff 	mov    -0x3c8(%rbp),%rax
  402a14:	48 05 e0 00 00 00    	add    $0xe0,%rax
  402a1a:	48 89 85 78 fd ff ff 	mov    %rax,-0x288(%rbp)
  402a21:	c5 f8 28 45 d0       	vmovaps -0x30(%rbp),%xmm0
  402a26:	c5 f8 29 85 60 fd ff 	vmovaps %xmm0,-0x2a0(%rbp)
  402a2d:	ff 
  402a2e:	48 8b 85 78 fd ff ff 	mov    -0x288(%rbp),%rax
  402a35:	c5 f8 28 85 60 fd ff 	vmovaps -0x2a0(%rbp),%xmm0
  402a3c:	ff 
  402a3d:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  402a41:	48 8b 85 38 fc ff ff 	mov    -0x3c8(%rbp),%rax
  402a48:	48 05 f0 00 00 00    	add    $0xf0,%rax
  402a4e:	48 89 85 58 fd ff ff 	mov    %rax,-0x2a8(%rbp)
  402a55:	c5 f8 28 45 c0       	vmovaps -0x40(%rbp),%xmm0
  402a5a:	c5 f8 29 85 40 fd ff 	vmovaps %xmm0,-0x2c0(%rbp)
  402a61:	ff 
  402a62:	48 8b 85 58 fd ff ff 	mov    -0x2a8(%rbp),%rax
  402a69:	c5 f8 28 85 40 fd ff 	vmovaps -0x2c0(%rbp),%xmm0
  402a70:	ff 
  402a71:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  402a75:	c9                   	leaveq 
  402a76:	c3                   	retq   

0000000000402a77 <_Z6kernelPKfS0_S0_PfS1_S1_iiS0_ii>:
  402a77:	55                   	push   %rbp
  402a78:	48 89 e5             	mov    %rsp,%rbp
  402a7b:	48 81 ec 38 0d 00 00 	sub    $0xd38,%rsp
  402a82:	48 89 bd 78 f2 ff ff 	mov    %rdi,-0xd88(%rbp)
  402a89:	48 89 b5 70 f2 ff ff 	mov    %rsi,-0xd90(%rbp)
  402a90:	48 89 95 68 f2 ff ff 	mov    %rdx,-0xd98(%rbp)
  402a97:	48 89 8d 60 f2 ff ff 	mov    %rcx,-0xda0(%rbp)
  402a9e:	4c 89 85 58 f2 ff ff 	mov    %r8,-0xda8(%rbp)
  402aa5:	4c 89 8d 50 f2 ff ff 	mov    %r9,-0xdb0(%rbp)
  402aac:	8b 45 30             	mov    0x30(%rbp),%eax
  402aaf:	0f af 45 10          	imul   0x10(%rbp),%eax
  402ab3:	8b 55 18             	mov    0x18(%rbp),%edx
  402ab6:	c1 e2 02             	shl    $0x2,%edx
  402ab9:	01 d0                	add    %edx,%eax
  402abb:	48 98                	cltq   
  402abd:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  402ac4:	00 
  402ac5:	48 8b 85 78 f2 ff ff 	mov    -0xd88(%rbp),%rax
  402acc:	48 01 d0             	add    %rdx,%rax
  402acf:	48 89 85 38 fe ff ff 	mov    %rax,-0x1c8(%rbp)
  402ad6:	48 8b 85 38 fe ff ff 	mov    -0x1c8(%rbp),%rax
  402add:	c4 e2 79 18 00       	vbroadcastss (%rax),%xmm0
  402ae2:	c5 f8 29 45 f0       	vmovaps %xmm0,-0x10(%rbp)
  402ae7:	8b 45 30             	mov    0x30(%rbp),%eax
  402aea:	0f af 45 10          	imul   0x10(%rbp),%eax
  402aee:	8b 55 18             	mov    0x18(%rbp),%edx
  402af1:	c1 e2 02             	shl    $0x2,%edx
  402af4:	01 d0                	add    %edx,%eax
  402af6:	48 98                	cltq   
  402af8:	48 83 c0 01          	add    $0x1,%rax
  402afc:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  402b03:	00 
  402b04:	48 8b 85 78 f2 ff ff 	mov    -0xd88(%rbp),%rax
  402b0b:	48 01 d0             	add    %rdx,%rax
  402b0e:	48 89 85 30 fe ff ff 	mov    %rax,-0x1d0(%rbp)
  402b15:	48 8b 85 30 fe ff ff 	mov    -0x1d0(%rbp),%rax
  402b1c:	c4 e2 79 18 00       	vbroadcastss (%rax),%xmm0
  402b21:	c5 f8 29 45 e0       	vmovaps %xmm0,-0x20(%rbp)
  402b26:	8b 45 30             	mov    0x30(%rbp),%eax
  402b29:	0f af 45 10          	imul   0x10(%rbp),%eax
  402b2d:	8b 55 18             	mov    0x18(%rbp),%edx
  402b30:	c1 e2 02             	shl    $0x2,%edx
  402b33:	01 d0                	add    %edx,%eax
  402b35:	48 98                	cltq   
  402b37:	48 83 c0 02          	add    $0x2,%rax
  402b3b:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  402b42:	00 
  402b43:	48 8b 85 78 f2 ff ff 	mov    -0xd88(%rbp),%rax
  402b4a:	48 01 d0             	add    %rdx,%rax
  402b4d:	48 89 85 28 fe ff ff 	mov    %rax,-0x1d8(%rbp)
  402b54:	48 8b 85 28 fe ff ff 	mov    -0x1d8(%rbp),%rax
  402b5b:	c4 e2 79 18 00       	vbroadcastss (%rax),%xmm0
  402b60:	c5 f8 29 45 d0       	vmovaps %xmm0,-0x30(%rbp)
  402b65:	8b 45 30             	mov    0x30(%rbp),%eax
  402b68:	0f af 45 10          	imul   0x10(%rbp),%eax
  402b6c:	8b 55 18             	mov    0x18(%rbp),%edx
  402b6f:	c1 e2 02             	shl    $0x2,%edx
  402b72:	01 d0                	add    %edx,%eax
  402b74:	48 98                	cltq   
  402b76:	48 83 c0 03          	add    $0x3,%rax
  402b7a:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  402b81:	00 
  402b82:	48 8b 85 78 f2 ff ff 	mov    -0xd88(%rbp),%rax
  402b89:	48 01 d0             	add    %rdx,%rax
  402b8c:	48 89 85 20 fe ff ff 	mov    %rax,-0x1e0(%rbp)
  402b93:	48 8b 85 20 fe ff ff 	mov    -0x1e0(%rbp),%rax
  402b9a:	c4 e2 79 18 00       	vbroadcastss (%rax),%xmm0
  402b9f:	c5 f8 29 45 c0       	vmovaps %xmm0,-0x40(%rbp)
  402ba4:	8b 45 30             	mov    0x30(%rbp),%eax
  402ba7:	0f af 45 10          	imul   0x10(%rbp),%eax
  402bab:	8b 55 18             	mov    0x18(%rbp),%edx
  402bae:	c1 e2 02             	shl    $0x2,%edx
  402bb1:	01 d0                	add    %edx,%eax
  402bb3:	48 98                	cltq   
  402bb5:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  402bbc:	00 
  402bbd:	48 8b 85 70 f2 ff ff 	mov    -0xd90(%rbp),%rax
  402bc4:	48 01 d0             	add    %rdx,%rax
  402bc7:	48 89 85 18 fe ff ff 	mov    %rax,-0x1e8(%rbp)
  402bce:	48 8b 85 18 fe ff ff 	mov    -0x1e8(%rbp),%rax
  402bd5:	c4 e2 79 18 00       	vbroadcastss (%rax),%xmm0
  402bda:	c5 f8 29 45 b0       	vmovaps %xmm0,-0x50(%rbp)
  402bdf:	8b 45 30             	mov    0x30(%rbp),%eax
  402be2:	0f af 45 10          	imul   0x10(%rbp),%eax
  402be6:	8b 55 18             	mov    0x18(%rbp),%edx
  402be9:	c1 e2 02             	shl    $0x2,%edx
  402bec:	01 d0                	add    %edx,%eax
  402bee:	48 98                	cltq   
  402bf0:	48 83 c0 01          	add    $0x1,%rax
  402bf4:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  402bfb:	00 
  402bfc:	48 8b 85 70 f2 ff ff 	mov    -0xd90(%rbp),%rax
  402c03:	48 01 d0             	add    %rdx,%rax
  402c06:	48 89 85 10 fe ff ff 	mov    %rax,-0x1f0(%rbp)
  402c0d:	48 8b 85 10 fe ff ff 	mov    -0x1f0(%rbp),%rax
  402c14:	c4 e2 79 18 00       	vbroadcastss (%rax),%xmm0
  402c19:	c5 f8 29 45 a0       	vmovaps %xmm0,-0x60(%rbp)
  402c1e:	8b 45 30             	mov    0x30(%rbp),%eax
  402c21:	0f af 45 10          	imul   0x10(%rbp),%eax
  402c25:	8b 55 18             	mov    0x18(%rbp),%edx
  402c28:	c1 e2 02             	shl    $0x2,%edx
  402c2b:	01 d0                	add    %edx,%eax
  402c2d:	48 98                	cltq   
  402c2f:	48 83 c0 02          	add    $0x2,%rax
  402c33:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  402c3a:	00 
  402c3b:	48 8b 85 70 f2 ff ff 	mov    -0xd90(%rbp),%rax
  402c42:	48 01 d0             	add    %rdx,%rax
  402c45:	48 89 85 08 fe ff ff 	mov    %rax,-0x1f8(%rbp)
  402c4c:	48 8b 85 08 fe ff ff 	mov    -0x1f8(%rbp),%rax
  402c53:	c4 e2 79 18 00       	vbroadcastss (%rax),%xmm0
  402c58:	c5 f8 29 45 90       	vmovaps %xmm0,-0x70(%rbp)
  402c5d:	8b 45 30             	mov    0x30(%rbp),%eax
  402c60:	0f af 45 10          	imul   0x10(%rbp),%eax
  402c64:	8b 55 18             	mov    0x18(%rbp),%edx
  402c67:	c1 e2 02             	shl    $0x2,%edx
  402c6a:	01 d0                	add    %edx,%eax
  402c6c:	48 98                	cltq   
  402c6e:	48 83 c0 03          	add    $0x3,%rax
  402c72:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  402c79:	00 
  402c7a:	48 8b 85 70 f2 ff ff 	mov    -0xd90(%rbp),%rax
  402c81:	48 01 d0             	add    %rdx,%rax
  402c84:	48 89 85 00 fe ff ff 	mov    %rax,-0x200(%rbp)
  402c8b:	48 8b 85 00 fe ff ff 	mov    -0x200(%rbp),%rax
  402c92:	c4 e2 79 18 00       	vbroadcastss (%rax),%xmm0
  402c97:	c5 f8 29 45 80       	vmovaps %xmm0,-0x80(%rbp)
  402c9c:	8b 45 30             	mov    0x30(%rbp),%eax
  402c9f:	0f af 45 10          	imul   0x10(%rbp),%eax
  402ca3:	8b 55 18             	mov    0x18(%rbp),%edx
  402ca6:	c1 e2 02             	shl    $0x2,%edx
  402ca9:	01 d0                	add    %edx,%eax
  402cab:	48 98                	cltq   
  402cad:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  402cb4:	00 
  402cb5:	48 8b 85 68 f2 ff ff 	mov    -0xd98(%rbp),%rax
  402cbc:	48 01 d0             	add    %rdx,%rax
  402cbf:	48 89 85 f8 fd ff ff 	mov    %rax,-0x208(%rbp)
  402cc6:	48 8b 85 f8 fd ff ff 	mov    -0x208(%rbp),%rax
  402ccd:	c4 e2 79 18 00       	vbroadcastss (%rax),%xmm0
  402cd2:	c5 f8 29 85 70 ff ff 	vmovaps %xmm0,-0x90(%rbp)
  402cd9:	ff 
  402cda:	8b 45 30             	mov    0x30(%rbp),%eax
  402cdd:	0f af 45 10          	imul   0x10(%rbp),%eax
  402ce1:	8b 55 18             	mov    0x18(%rbp),%edx
  402ce4:	c1 e2 02             	shl    $0x2,%edx
  402ce7:	01 d0                	add    %edx,%eax
  402ce9:	48 98                	cltq   
  402ceb:	48 83 c0 01          	add    $0x1,%rax
  402cef:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  402cf6:	00 
  402cf7:	48 8b 85 68 f2 ff ff 	mov    -0xd98(%rbp),%rax
  402cfe:	48 01 d0             	add    %rdx,%rax
  402d01:	48 89 85 f0 fd ff ff 	mov    %rax,-0x210(%rbp)
  402d08:	48 8b 85 f0 fd ff ff 	mov    -0x210(%rbp),%rax
  402d0f:	c4 e2 79 18 00       	vbroadcastss (%rax),%xmm0
  402d14:	c5 f8 29 85 60 ff ff 	vmovaps %xmm0,-0xa0(%rbp)
  402d1b:	ff 
  402d1c:	8b 45 30             	mov    0x30(%rbp),%eax
  402d1f:	0f af 45 10          	imul   0x10(%rbp),%eax
  402d23:	8b 55 18             	mov    0x18(%rbp),%edx
  402d26:	c1 e2 02             	shl    $0x2,%edx
  402d29:	01 d0                	add    %edx,%eax
  402d2b:	48 98                	cltq   
  402d2d:	48 83 c0 02          	add    $0x2,%rax
  402d31:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  402d38:	00 
  402d39:	48 8b 85 68 f2 ff ff 	mov    -0xd98(%rbp),%rax
  402d40:	48 01 d0             	add    %rdx,%rax
  402d43:	48 89 85 e8 fd ff ff 	mov    %rax,-0x218(%rbp)
  402d4a:	48 8b 85 e8 fd ff ff 	mov    -0x218(%rbp),%rax
  402d51:	c4 e2 79 18 00       	vbroadcastss (%rax),%xmm0
  402d56:	c5 f8 29 85 50 ff ff 	vmovaps %xmm0,-0xb0(%rbp)
  402d5d:	ff 
  402d5e:	8b 45 30             	mov    0x30(%rbp),%eax
  402d61:	0f af 45 10          	imul   0x10(%rbp),%eax
  402d65:	8b 55 18             	mov    0x18(%rbp),%edx
  402d68:	c1 e2 02             	shl    $0x2,%edx
  402d6b:	01 d0                	add    %edx,%eax
  402d6d:	48 98                	cltq   
  402d6f:	48 83 c0 03          	add    $0x3,%rax
  402d73:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  402d7a:	00 
  402d7b:	48 8b 85 68 f2 ff ff 	mov    -0xd98(%rbp),%rax
  402d82:	48 01 d0             	add    %rdx,%rax
  402d85:	48 89 85 e0 fd ff ff 	mov    %rax,-0x220(%rbp)
  402d8c:	48 8b 85 e0 fd ff ff 	mov    -0x220(%rbp),%rax
  402d93:	c4 e2 79 18 00       	vbroadcastss (%rax),%xmm0
  402d98:	c5 f8 29 85 40 ff ff 	vmovaps %xmm0,-0xc0(%rbp)
  402d9f:	ff 
  402da0:	8b 45 30             	mov    0x30(%rbp),%eax
  402da3:	48 63 d0             	movslq %eax,%rdx
  402da6:	8b 45 10             	mov    0x10(%rbp),%eax
  402da9:	48 98                	cltq   
  402dab:	48 0f af d0          	imul   %rax,%rdx
  402daf:	8b 45 18             	mov    0x18(%rbp),%eax
  402db2:	48 98                	cltq   
  402db4:	48 01 d0             	add    %rdx,%rax
  402db7:	48 c1 e0 04          	shl    $0x4,%rax
  402dbb:	48 89 c2             	mov    %rax,%rdx
  402dbe:	48 8b 85 60 f2 ff ff 	mov    -0xda0(%rbp),%rax
  402dc5:	48 01 d0             	add    %rdx,%rax
  402dc8:	48 89 85 d8 fd ff ff 	mov    %rax,-0x228(%rbp)
  402dcf:	48 8b 85 d8 fd ff ff 	mov    -0x228(%rbp),%rax
  402dd6:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  402dda:	c5 f8 29 85 30 ff ff 	vmovaps %xmm0,-0xd0(%rbp)
  402de1:	ff 
  402de2:	8b 45 30             	mov    0x30(%rbp),%eax
  402de5:	48 63 d0             	movslq %eax,%rdx
  402de8:	8b 45 10             	mov    0x10(%rbp),%eax
  402deb:	48 98                	cltq   
  402ded:	48 0f af d0          	imul   %rax,%rdx
  402df1:	8b 45 18             	mov    0x18(%rbp),%eax
  402df4:	48 98                	cltq   
  402df6:	48 01 d0             	add    %rdx,%rax
  402df9:	48 c1 e0 04          	shl    $0x4,%rax
  402dfd:	48 89 c2             	mov    %rax,%rdx
  402e00:	48 8b 85 58 f2 ff ff 	mov    -0xda8(%rbp),%rax
  402e07:	48 01 d0             	add    %rdx,%rax
  402e0a:	48 89 85 d0 fd ff ff 	mov    %rax,-0x230(%rbp)
  402e11:	48 8b 85 d0 fd ff ff 	mov    -0x230(%rbp),%rax
  402e18:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  402e1c:	c5 f8 29 85 20 ff ff 	vmovaps %xmm0,-0xe0(%rbp)
  402e23:	ff 
  402e24:	8b 45 30             	mov    0x30(%rbp),%eax
  402e27:	48 63 d0             	movslq %eax,%rdx
  402e2a:	8b 45 10             	mov    0x10(%rbp),%eax
  402e2d:	48 98                	cltq   
  402e2f:	48 0f af d0          	imul   %rax,%rdx
  402e33:	8b 45 18             	mov    0x18(%rbp),%eax
  402e36:	48 98                	cltq   
  402e38:	48 01 d0             	add    %rdx,%rax
  402e3b:	48 c1 e0 04          	shl    $0x4,%rax
  402e3f:	48 89 c2             	mov    %rax,%rdx
  402e42:	48 8b 85 50 f2 ff ff 	mov    -0xdb0(%rbp),%rax
  402e49:	48 01 d0             	add    %rdx,%rax
  402e4c:	48 89 85 c8 fd ff ff 	mov    %rax,-0x238(%rbp)
  402e53:	48 8b 85 c8 fd ff ff 	mov    -0x238(%rbp),%rax
  402e5a:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  402e5e:	c5 f8 29 85 10 ff ff 	vmovaps %xmm0,-0xf0(%rbp)
  402e65:	ff 
  402e66:	8b 45 30             	mov    0x30(%rbp),%eax
  402e69:	c1 e0 02             	shl    $0x2,%eax
  402e6c:	0f af 45 10          	imul   0x10(%rbp),%eax
  402e70:	8b 55 18             	mov    0x18(%rbp),%edx
  402e73:	c1 e2 02             	shl    $0x2,%edx
  402e76:	01 c2                	add    %eax,%edx
  402e78:	8b 45 30             	mov    0x30(%rbp),%eax
  402e7b:	01 d0                	add    %edx,%eax
  402e7d:	48 98                	cltq   
  402e7f:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  402e86:	00 
  402e87:	48 8b 85 60 f2 ff ff 	mov    -0xda0(%rbp),%rax
  402e8e:	48 01 d0             	add    %rdx,%rax
  402e91:	48 89 85 c0 fd ff ff 	mov    %rax,-0x240(%rbp)
  402e98:	48 8b 85 c0 fd ff ff 	mov    -0x240(%rbp),%rax
  402e9f:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  402ea3:	c5 f8 29 85 00 ff ff 	vmovaps %xmm0,-0x100(%rbp)
  402eaa:	ff 
  402eab:	8b 45 30             	mov    0x30(%rbp),%eax
  402eae:	c1 e0 02             	shl    $0x2,%eax
  402eb1:	0f af 45 10          	imul   0x10(%rbp),%eax
  402eb5:	8b 55 18             	mov    0x18(%rbp),%edx
  402eb8:	c1 e2 02             	shl    $0x2,%edx
  402ebb:	01 c2                	add    %eax,%edx
  402ebd:	8b 45 30             	mov    0x30(%rbp),%eax
  402ec0:	01 d0                	add    %edx,%eax
  402ec2:	48 98                	cltq   
  402ec4:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  402ecb:	00 
  402ecc:	48 8b 85 58 f2 ff ff 	mov    -0xda8(%rbp),%rax
  402ed3:	48 01 d0             	add    %rdx,%rax
  402ed6:	48 89 85 b8 fd ff ff 	mov    %rax,-0x248(%rbp)
  402edd:	48 8b 85 b8 fd ff ff 	mov    -0x248(%rbp),%rax
  402ee4:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  402ee8:	c5 f8 29 85 f0 fe ff 	vmovaps %xmm0,-0x110(%rbp)
  402eef:	ff 
  402ef0:	8b 45 30             	mov    0x30(%rbp),%eax
  402ef3:	c1 e0 02             	shl    $0x2,%eax
  402ef6:	0f af 45 10          	imul   0x10(%rbp),%eax
  402efa:	8b 55 18             	mov    0x18(%rbp),%edx
  402efd:	c1 e2 02             	shl    $0x2,%edx
  402f00:	01 c2                	add    %eax,%edx
  402f02:	8b 45 30             	mov    0x30(%rbp),%eax
  402f05:	01 d0                	add    %edx,%eax
  402f07:	48 98                	cltq   
  402f09:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  402f10:	00 
  402f11:	48 8b 85 50 f2 ff ff 	mov    -0xdb0(%rbp),%rax
  402f18:	48 01 d0             	add    %rdx,%rax
  402f1b:	48 89 85 b0 fd ff ff 	mov    %rax,-0x250(%rbp)
  402f22:	48 8b 85 b0 fd ff ff 	mov    -0x250(%rbp),%rax
  402f29:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  402f2d:	c5 f8 29 85 e0 fe ff 	vmovaps %xmm0,-0x120(%rbp)
  402f34:	ff 
  402f35:	8b 45 30             	mov    0x30(%rbp),%eax
  402f38:	c1 e0 02             	shl    $0x2,%eax
  402f3b:	0f af 45 10          	imul   0x10(%rbp),%eax
  402f3f:	8b 55 18             	mov    0x18(%rbp),%edx
  402f42:	c1 e2 02             	shl    $0x2,%edx
  402f45:	01 c2                	add    %eax,%edx
  402f47:	8b 45 30             	mov    0x30(%rbp),%eax
  402f4a:	01 c0                	add    %eax,%eax
  402f4c:	01 d0                	add    %edx,%eax
  402f4e:	48 98                	cltq   
  402f50:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  402f57:	00 
  402f58:	48 8b 85 60 f2 ff ff 	mov    -0xda0(%rbp),%rax
  402f5f:	48 01 d0             	add    %rdx,%rax
  402f62:	48 89 85 a8 fd ff ff 	mov    %rax,-0x258(%rbp)
  402f69:	48 8b 85 a8 fd ff ff 	mov    -0x258(%rbp),%rax
  402f70:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  402f74:	c5 f8 29 85 d0 fe ff 	vmovaps %xmm0,-0x130(%rbp)
  402f7b:	ff 
  402f7c:	8b 45 30             	mov    0x30(%rbp),%eax
  402f7f:	c1 e0 02             	shl    $0x2,%eax
  402f82:	0f af 45 10          	imul   0x10(%rbp),%eax
  402f86:	8b 55 18             	mov    0x18(%rbp),%edx
  402f89:	c1 e2 02             	shl    $0x2,%edx
  402f8c:	01 c2                	add    %eax,%edx
  402f8e:	8b 45 30             	mov    0x30(%rbp),%eax
  402f91:	01 c0                	add    %eax,%eax
  402f93:	01 d0                	add    %edx,%eax
  402f95:	48 98                	cltq   
  402f97:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  402f9e:	00 
  402f9f:	48 8b 85 58 f2 ff ff 	mov    -0xda8(%rbp),%rax
  402fa6:	48 01 d0             	add    %rdx,%rax
  402fa9:	48 89 85 a0 fd ff ff 	mov    %rax,-0x260(%rbp)
  402fb0:	48 8b 85 a0 fd ff ff 	mov    -0x260(%rbp),%rax
  402fb7:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  402fbb:	c5 f8 29 85 c0 fe ff 	vmovaps %xmm0,-0x140(%rbp)
  402fc2:	ff 
  402fc3:	8b 45 30             	mov    0x30(%rbp),%eax
  402fc6:	c1 e0 02             	shl    $0x2,%eax
  402fc9:	0f af 45 10          	imul   0x10(%rbp),%eax
  402fcd:	8b 55 18             	mov    0x18(%rbp),%edx
  402fd0:	c1 e2 02             	shl    $0x2,%edx
  402fd3:	01 c2                	add    %eax,%edx
  402fd5:	8b 45 30             	mov    0x30(%rbp),%eax
  402fd8:	01 c0                	add    %eax,%eax
  402fda:	01 d0                	add    %edx,%eax
  402fdc:	48 98                	cltq   
  402fde:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  402fe5:	00 
  402fe6:	48 8b 85 50 f2 ff ff 	mov    -0xdb0(%rbp),%rax
  402fed:	48 01 d0             	add    %rdx,%rax
  402ff0:	48 89 85 98 fd ff ff 	mov    %rax,-0x268(%rbp)
  402ff7:	48 8b 85 98 fd ff ff 	mov    -0x268(%rbp),%rax
  402ffe:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  403002:	c5 f8 29 85 b0 fe ff 	vmovaps %xmm0,-0x150(%rbp)
  403009:	ff 
  40300a:	8b 45 30             	mov    0x30(%rbp),%eax
  40300d:	c1 e0 02             	shl    $0x2,%eax
  403010:	0f af 45 10          	imul   0x10(%rbp),%eax
  403014:	8b 55 18             	mov    0x18(%rbp),%edx
  403017:	c1 e2 02             	shl    $0x2,%edx
  40301a:	8d 0c 10             	lea    (%rax,%rdx,1),%ecx
  40301d:	8b 55 30             	mov    0x30(%rbp),%edx
  403020:	89 d0                	mov    %edx,%eax
  403022:	01 c0                	add    %eax,%eax
  403024:	01 d0                	add    %edx,%eax
  403026:	01 c8                	add    %ecx,%eax
  403028:	48 98                	cltq   
  40302a:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  403031:	00 
  403032:	48 8b 85 60 f2 ff ff 	mov    -0xda0(%rbp),%rax
  403039:	48 01 d0             	add    %rdx,%rax
  40303c:	48 89 85 90 fd ff ff 	mov    %rax,-0x270(%rbp)
  403043:	48 8b 85 90 fd ff ff 	mov    -0x270(%rbp),%rax
  40304a:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  40304e:	c5 f8 29 85 a0 fe ff 	vmovaps %xmm0,-0x160(%rbp)
  403055:	ff 
  403056:	8b 45 30             	mov    0x30(%rbp),%eax
  403059:	c1 e0 02             	shl    $0x2,%eax
  40305c:	0f af 45 10          	imul   0x10(%rbp),%eax
  403060:	8b 55 18             	mov    0x18(%rbp),%edx
  403063:	c1 e2 02             	shl    $0x2,%edx
  403066:	8d 0c 10             	lea    (%rax,%rdx,1),%ecx
  403069:	8b 55 30             	mov    0x30(%rbp),%edx
  40306c:	89 d0                	mov    %edx,%eax
  40306e:	01 c0                	add    %eax,%eax
  403070:	01 d0                	add    %edx,%eax
  403072:	01 c8                	add    %ecx,%eax
  403074:	48 98                	cltq   
  403076:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  40307d:	00 
  40307e:	48 8b 85 58 f2 ff ff 	mov    -0xda8(%rbp),%rax
  403085:	48 01 d0             	add    %rdx,%rax
  403088:	48 89 85 88 fd ff ff 	mov    %rax,-0x278(%rbp)
  40308f:	48 8b 85 88 fd ff ff 	mov    -0x278(%rbp),%rax
  403096:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  40309a:	c5 f8 29 85 90 fe ff 	vmovaps %xmm0,-0x170(%rbp)
  4030a1:	ff 
  4030a2:	8b 45 30             	mov    0x30(%rbp),%eax
  4030a5:	c1 e0 02             	shl    $0x2,%eax
  4030a8:	0f af 45 10          	imul   0x10(%rbp),%eax
  4030ac:	8b 55 18             	mov    0x18(%rbp),%edx
  4030af:	c1 e2 02             	shl    $0x2,%edx
  4030b2:	8d 0c 10             	lea    (%rax,%rdx,1),%ecx
  4030b5:	8b 55 30             	mov    0x30(%rbp),%edx
  4030b8:	89 d0                	mov    %edx,%eax
  4030ba:	01 c0                	add    %eax,%eax
  4030bc:	01 d0                	add    %edx,%eax
  4030be:	01 c8                	add    %ecx,%eax
  4030c0:	48 98                	cltq   
  4030c2:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  4030c9:	00 
  4030ca:	48 8b 85 50 f2 ff ff 	mov    -0xdb0(%rbp),%rax
  4030d1:	48 01 d0             	add    %rdx,%rax
  4030d4:	48 89 85 80 fd ff ff 	mov    %rax,-0x280(%rbp)
  4030db:	48 8b 85 80 fd ff ff 	mov    -0x280(%rbp),%rax
  4030e2:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  4030e6:	c5 f8 29 85 80 fe ff 	vmovaps %xmm0,-0x180(%rbp)
  4030ed:	ff 
  4030ee:	48 8b 45 20          	mov    0x20(%rbp),%rax
  4030f2:	48 89 85 78 fd ff ff 	mov    %rax,-0x288(%rbp)
  4030f9:	48 8b 85 78 fd ff ff 	mov    -0x288(%rbp),%rax
  403100:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  403104:	c5 f8 29 85 70 fe ff 	vmovaps %xmm0,-0x190(%rbp)
  40310b:	ff 
  40310c:	48 8b 45 20          	mov    0x20(%rbp),%rax
  403110:	48 83 c0 10          	add    $0x10,%rax
  403114:	48 89 85 70 fd ff ff 	mov    %rax,-0x290(%rbp)
  40311b:	48 8b 85 70 fd ff ff 	mov    -0x290(%rbp),%rax
  403122:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  403126:	c5 f8 29 85 60 fe ff 	vmovaps %xmm0,-0x1a0(%rbp)
  40312d:	ff 
  40312e:	48 8b 45 20          	mov    0x20(%rbp),%rax
  403132:	48 83 c0 20          	add    $0x20,%rax
  403136:	48 89 85 68 fd ff ff 	mov    %rax,-0x298(%rbp)
  40313d:	48 8b 85 68 fd ff ff 	mov    -0x298(%rbp),%rax
  403144:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  403148:	c5 f8 29 85 50 fe ff 	vmovaps %xmm0,-0x1b0(%rbp)
  40314f:	ff 
  403150:	48 8b 45 20          	mov    0x20(%rbp),%rax
  403154:	48 83 c0 30          	add    $0x30,%rax
  403158:	48 89 85 60 fd ff ff 	mov    %rax,-0x2a0(%rbp)
  40315f:	48 8b 85 60 fd ff ff 	mov    -0x2a0(%rbp),%rax
  403166:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  40316a:	c5 f8 29 85 40 fe ff 	vmovaps %xmm0,-0x1c0(%rbp)
  403171:	ff 
  403172:	c5 f8 28 45 f0       	vmovaps -0x10(%rbp),%xmm0
  403177:	c5 f8 29 85 50 fd ff 	vmovaps %xmm0,-0x2b0(%rbp)
  40317e:	ff 
  40317f:	c5 f8 28 85 70 fe ff 	vmovaps -0x190(%rbp),%xmm0
  403186:	ff 
  403187:	c5 f8 29 85 40 fd ff 	vmovaps %xmm0,-0x2c0(%rbp)
  40318e:	ff 
  40318f:	c5 f8 28 85 30 ff ff 	vmovaps -0xd0(%rbp),%xmm0
  403196:	ff 
  403197:	c5 f8 29 85 30 fd ff 	vmovaps %xmm0,-0x2d0(%rbp)
  40319e:	ff 
  40319f:	c5 f8 28 8d 40 fd ff 	vmovaps -0x2c0(%rbp),%xmm1
  4031a6:	ff 
  4031a7:	c5 f8 28 85 30 fd ff 	vmovaps -0x2d0(%rbp),%xmm0
  4031ae:	ff 
  4031af:	c4 e2 71 b8 85 50 fd 	vfmadd231ps -0x2b0(%rbp),%xmm1,%xmm0
  4031b6:	ff ff 
  4031b8:	c5 f8 29 85 30 ff ff 	vmovaps %xmm0,-0xd0(%rbp)
  4031bf:	ff 
  4031c0:	c5 f8 28 45 e0       	vmovaps -0x20(%rbp),%xmm0
  4031c5:	c5 f8 29 85 20 fd ff 	vmovaps %xmm0,-0x2e0(%rbp)
  4031cc:	ff 
  4031cd:	c5 f8 28 85 60 fe ff 	vmovaps -0x1a0(%rbp),%xmm0
  4031d4:	ff 
  4031d5:	c5 f8 29 85 10 fd ff 	vmovaps %xmm0,-0x2f0(%rbp)
  4031dc:	ff 
  4031dd:	c5 f8 28 85 30 ff ff 	vmovaps -0xd0(%rbp),%xmm0
  4031e4:	ff 
  4031e5:	c5 f8 29 85 00 fd ff 	vmovaps %xmm0,-0x300(%rbp)
  4031ec:	ff 
  4031ed:	c5 f8 28 8d 10 fd ff 	vmovaps -0x2f0(%rbp),%xmm1
  4031f4:	ff 
  4031f5:	c5 f8 28 85 00 fd ff 	vmovaps -0x300(%rbp),%xmm0
  4031fc:	ff 
  4031fd:	c4 e2 71 b8 85 20 fd 	vfmadd231ps -0x2e0(%rbp),%xmm1,%xmm0
  403204:	ff ff 
  403206:	c5 f8 29 85 30 ff ff 	vmovaps %xmm0,-0xd0(%rbp)
  40320d:	ff 
  40320e:	c5 f8 28 45 d0       	vmovaps -0x30(%rbp),%xmm0
  403213:	c5 f8 29 85 f0 fc ff 	vmovaps %xmm0,-0x310(%rbp)
  40321a:	ff 
  40321b:	c5 f8 28 85 50 fe ff 	vmovaps -0x1b0(%rbp),%xmm0
  403222:	ff 
  403223:	c5 f8 29 85 e0 fc ff 	vmovaps %xmm0,-0x320(%rbp)
  40322a:	ff 
  40322b:	c5 f8 28 85 30 ff ff 	vmovaps -0xd0(%rbp),%xmm0
  403232:	ff 
  403233:	c5 f8 29 85 d0 fc ff 	vmovaps %xmm0,-0x330(%rbp)
  40323a:	ff 
  40323b:	c5 f8 28 8d e0 fc ff 	vmovaps -0x320(%rbp),%xmm1
  403242:	ff 
  403243:	c5 f8 28 85 d0 fc ff 	vmovaps -0x330(%rbp),%xmm0
  40324a:	ff 
  40324b:	c4 e2 71 b8 85 f0 fc 	vfmadd231ps -0x310(%rbp),%xmm1,%xmm0
  403252:	ff ff 
  403254:	c5 f8 29 85 30 ff ff 	vmovaps %xmm0,-0xd0(%rbp)
  40325b:	ff 
  40325c:	c5 f8 28 45 c0       	vmovaps -0x40(%rbp),%xmm0
  403261:	c5 f8 29 85 c0 fc ff 	vmovaps %xmm0,-0x340(%rbp)
  403268:	ff 
  403269:	c5 f8 28 85 40 fe ff 	vmovaps -0x1c0(%rbp),%xmm0
  403270:	ff 
  403271:	c5 f8 29 85 b0 fc ff 	vmovaps %xmm0,-0x350(%rbp)
  403278:	ff 
  403279:	c5 f8 28 85 30 ff ff 	vmovaps -0xd0(%rbp),%xmm0
  403280:	ff 
  403281:	c5 f8 29 85 a0 fc ff 	vmovaps %xmm0,-0x360(%rbp)
  403288:	ff 
  403289:	c5 f8 28 8d b0 fc ff 	vmovaps -0x350(%rbp),%xmm1
  403290:	ff 
  403291:	c5 f8 28 85 a0 fc ff 	vmovaps -0x360(%rbp),%xmm0
  403298:	ff 
  403299:	c4 e2 71 b8 85 c0 fc 	vfmadd231ps -0x340(%rbp),%xmm1,%xmm0
  4032a0:	ff ff 
  4032a2:	c5 f8 29 85 30 ff ff 	vmovaps %xmm0,-0xd0(%rbp)
  4032a9:	ff 
  4032aa:	c5 f8 28 45 b0       	vmovaps -0x50(%rbp),%xmm0
  4032af:	c5 f8 29 85 90 fc ff 	vmovaps %xmm0,-0x370(%rbp)
  4032b6:	ff 
  4032b7:	c5 f8 28 85 70 fe ff 	vmovaps -0x190(%rbp),%xmm0
  4032be:	ff 
  4032bf:	c5 f8 29 85 80 fc ff 	vmovaps %xmm0,-0x380(%rbp)
  4032c6:	ff 
  4032c7:	c5 f8 28 85 20 ff ff 	vmovaps -0xe0(%rbp),%xmm0
  4032ce:	ff 
  4032cf:	c5 f8 29 85 70 fc ff 	vmovaps %xmm0,-0x390(%rbp)
  4032d6:	ff 
  4032d7:	c5 f8 28 8d 80 fc ff 	vmovaps -0x380(%rbp),%xmm1
  4032de:	ff 
  4032df:	c5 f8 28 85 70 fc ff 	vmovaps -0x390(%rbp),%xmm0
  4032e6:	ff 
  4032e7:	c4 e2 71 b8 85 90 fc 	vfmadd231ps -0x370(%rbp),%xmm1,%xmm0
  4032ee:	ff ff 
  4032f0:	c5 f8 29 85 20 ff ff 	vmovaps %xmm0,-0xe0(%rbp)
  4032f7:	ff 
  4032f8:	c5 f8 28 45 a0       	vmovaps -0x60(%rbp),%xmm0
  4032fd:	c5 f8 29 85 60 fc ff 	vmovaps %xmm0,-0x3a0(%rbp)
  403304:	ff 
  403305:	c5 f8 28 85 60 fe ff 	vmovaps -0x1a0(%rbp),%xmm0
  40330c:	ff 
  40330d:	c5 f8 29 85 50 fc ff 	vmovaps %xmm0,-0x3b0(%rbp)
  403314:	ff 
  403315:	c5 f8 28 85 20 ff ff 	vmovaps -0xe0(%rbp),%xmm0
  40331c:	ff 
  40331d:	c5 f8 29 85 40 fc ff 	vmovaps %xmm0,-0x3c0(%rbp)
  403324:	ff 
  403325:	c5 f8 28 8d 50 fc ff 	vmovaps -0x3b0(%rbp),%xmm1
  40332c:	ff 
  40332d:	c5 f8 28 85 40 fc ff 	vmovaps -0x3c0(%rbp),%xmm0
  403334:	ff 
  403335:	c4 e2 71 b8 85 60 fc 	vfmadd231ps -0x3a0(%rbp),%xmm1,%xmm0
  40333c:	ff ff 
  40333e:	c5 f8 29 85 20 ff ff 	vmovaps %xmm0,-0xe0(%rbp)
  403345:	ff 
  403346:	c5 f8 28 45 90       	vmovaps -0x70(%rbp),%xmm0
  40334b:	c5 f8 29 85 30 fc ff 	vmovaps %xmm0,-0x3d0(%rbp)
  403352:	ff 
  403353:	c5 f8 28 85 50 fe ff 	vmovaps -0x1b0(%rbp),%xmm0
  40335a:	ff 
  40335b:	c5 f8 29 85 20 fc ff 	vmovaps %xmm0,-0x3e0(%rbp)
  403362:	ff 
  403363:	c5 f8 28 85 20 ff ff 	vmovaps -0xe0(%rbp),%xmm0
  40336a:	ff 
  40336b:	c5 f8 29 85 10 fc ff 	vmovaps %xmm0,-0x3f0(%rbp)
  403372:	ff 
  403373:	c5 f8 28 8d 20 fc ff 	vmovaps -0x3e0(%rbp),%xmm1
  40337a:	ff 
  40337b:	c5 f8 28 85 10 fc ff 	vmovaps -0x3f0(%rbp),%xmm0
  403382:	ff 
  403383:	c4 e2 71 b8 85 30 fc 	vfmadd231ps -0x3d0(%rbp),%xmm1,%xmm0
  40338a:	ff ff 
  40338c:	c5 f8 29 85 20 ff ff 	vmovaps %xmm0,-0xe0(%rbp)
  403393:	ff 
  403394:	c5 f8 28 45 80       	vmovaps -0x80(%rbp),%xmm0
  403399:	c5 f8 29 85 00 fc ff 	vmovaps %xmm0,-0x400(%rbp)
  4033a0:	ff 
  4033a1:	c5 f8 28 85 40 fe ff 	vmovaps -0x1c0(%rbp),%xmm0
  4033a8:	ff 
  4033a9:	c5 f8 29 85 f0 fb ff 	vmovaps %xmm0,-0x410(%rbp)
  4033b0:	ff 
  4033b1:	c5 f8 28 85 20 ff ff 	vmovaps -0xe0(%rbp),%xmm0
  4033b8:	ff 
  4033b9:	c5 f8 29 85 e0 fb ff 	vmovaps %xmm0,-0x420(%rbp)
  4033c0:	ff 
  4033c1:	c5 f8 28 8d f0 fb ff 	vmovaps -0x410(%rbp),%xmm1
  4033c8:	ff 
  4033c9:	c5 f8 28 85 e0 fb ff 	vmovaps -0x420(%rbp),%xmm0
  4033d0:	ff 
  4033d1:	c4 e2 71 b8 85 00 fc 	vfmadd231ps -0x400(%rbp),%xmm1,%xmm0
  4033d8:	ff ff 
  4033da:	c5 f8 29 85 20 ff ff 	vmovaps %xmm0,-0xe0(%rbp)
  4033e1:	ff 
  4033e2:	c5 f8 28 85 70 ff ff 	vmovaps -0x90(%rbp),%xmm0
  4033e9:	ff 
  4033ea:	c5 f8 29 85 d0 fb ff 	vmovaps %xmm0,-0x430(%rbp)
  4033f1:	ff 
  4033f2:	c5 f8 28 85 70 fe ff 	vmovaps -0x190(%rbp),%xmm0
  4033f9:	ff 
  4033fa:	c5 f8 29 85 c0 fb ff 	vmovaps %xmm0,-0x440(%rbp)
  403401:	ff 
  403402:	c5 f8 28 85 10 ff ff 	vmovaps -0xf0(%rbp),%xmm0
  403409:	ff 
  40340a:	c5 f8 29 85 b0 fb ff 	vmovaps %xmm0,-0x450(%rbp)
  403411:	ff 
  403412:	c5 f8 28 8d c0 fb ff 	vmovaps -0x440(%rbp),%xmm1
  403419:	ff 
  40341a:	c5 f8 28 85 b0 fb ff 	vmovaps -0x450(%rbp),%xmm0
  403421:	ff 
  403422:	c4 e2 71 b8 85 d0 fb 	vfmadd231ps -0x430(%rbp),%xmm1,%xmm0
  403429:	ff ff 
  40342b:	c5 f8 29 85 10 ff ff 	vmovaps %xmm0,-0xf0(%rbp)
  403432:	ff 
  403433:	c5 f8 28 85 60 ff ff 	vmovaps -0xa0(%rbp),%xmm0
  40343a:	ff 
  40343b:	c5 f8 29 85 a0 fb ff 	vmovaps %xmm0,-0x460(%rbp)
  403442:	ff 
  403443:	c5 f8 28 85 60 fe ff 	vmovaps -0x1a0(%rbp),%xmm0
  40344a:	ff 
  40344b:	c5 f8 29 85 90 fb ff 	vmovaps %xmm0,-0x470(%rbp)
  403452:	ff 
  403453:	c5 f8 28 85 10 ff ff 	vmovaps -0xf0(%rbp),%xmm0
  40345a:	ff 
  40345b:	c5 f8 29 85 80 fb ff 	vmovaps %xmm0,-0x480(%rbp)
  403462:	ff 
  403463:	c5 f8 28 8d 90 fb ff 	vmovaps -0x470(%rbp),%xmm1
  40346a:	ff 
  40346b:	c5 f8 28 85 80 fb ff 	vmovaps -0x480(%rbp),%xmm0
  403472:	ff 
  403473:	c4 e2 71 b8 85 a0 fb 	vfmadd231ps -0x460(%rbp),%xmm1,%xmm0
  40347a:	ff ff 
  40347c:	c5 f8 29 85 10 ff ff 	vmovaps %xmm0,-0xf0(%rbp)
  403483:	ff 
  403484:	c5 f8 28 85 50 ff ff 	vmovaps -0xb0(%rbp),%xmm0
  40348b:	ff 
  40348c:	c5 f8 29 85 70 fb ff 	vmovaps %xmm0,-0x490(%rbp)
  403493:	ff 
  403494:	c5 f8 28 85 50 fe ff 	vmovaps -0x1b0(%rbp),%xmm0
  40349b:	ff 
  40349c:	c5 f8 29 85 60 fb ff 	vmovaps %xmm0,-0x4a0(%rbp)
  4034a3:	ff 
  4034a4:	c5 f8 28 85 10 ff ff 	vmovaps -0xf0(%rbp),%xmm0
  4034ab:	ff 
  4034ac:	c5 f8 29 85 50 fb ff 	vmovaps %xmm0,-0x4b0(%rbp)
  4034b3:	ff 
  4034b4:	c5 f8 28 8d 60 fb ff 	vmovaps -0x4a0(%rbp),%xmm1
  4034bb:	ff 
  4034bc:	c5 f8 28 85 50 fb ff 	vmovaps -0x4b0(%rbp),%xmm0
  4034c3:	ff 
  4034c4:	c4 e2 71 b8 85 70 fb 	vfmadd231ps -0x490(%rbp),%xmm1,%xmm0
  4034cb:	ff ff 
  4034cd:	c5 f8 29 85 10 ff ff 	vmovaps %xmm0,-0xf0(%rbp)
  4034d4:	ff 
  4034d5:	c5 f8 28 85 40 ff ff 	vmovaps -0xc0(%rbp),%xmm0
  4034dc:	ff 
  4034dd:	c5 f8 29 85 40 fb ff 	vmovaps %xmm0,-0x4c0(%rbp)
  4034e4:	ff 
  4034e5:	c5 f8 28 85 40 fe ff 	vmovaps -0x1c0(%rbp),%xmm0
  4034ec:	ff 
  4034ed:	c5 f8 29 85 30 fb ff 	vmovaps %xmm0,-0x4d0(%rbp)
  4034f4:	ff 
  4034f5:	c5 f8 28 85 10 ff ff 	vmovaps -0xf0(%rbp),%xmm0
  4034fc:	ff 
  4034fd:	c5 f8 29 85 20 fb ff 	vmovaps %xmm0,-0x4e0(%rbp)
  403504:	ff 
  403505:	c5 f8 28 8d 30 fb ff 	vmovaps -0x4d0(%rbp),%xmm1
  40350c:	ff 
  40350d:	c5 f8 28 85 20 fb ff 	vmovaps -0x4e0(%rbp),%xmm0
  403514:	ff 
  403515:	c4 e2 71 b8 85 40 fb 	vfmadd231ps -0x4c0(%rbp),%xmm1,%xmm0
  40351c:	ff ff 
  40351e:	c5 f8 29 85 10 ff ff 	vmovaps %xmm0,-0xf0(%rbp)
  403525:	ff 
  403526:	8b 45 30             	mov    0x30(%rbp),%eax
  403529:	48 63 d0             	movslq %eax,%rdx
  40352c:	8b 45 10             	mov    0x10(%rbp),%eax
  40352f:	48 98                	cltq   
  403531:	48 0f af d0          	imul   %rax,%rdx
  403535:	8b 45 18             	mov    0x18(%rbp),%eax
  403538:	48 98                	cltq   
  40353a:	48 01 d0             	add    %rdx,%rax
  40353d:	48 c1 e0 04          	shl    $0x4,%rax
  403541:	48 89 c2             	mov    %rax,%rdx
  403544:	48 8b 85 60 f2 ff ff 	mov    -0xda0(%rbp),%rax
  40354b:	48 01 d0             	add    %rdx,%rax
  40354e:	48 89 85 18 fb ff ff 	mov    %rax,-0x4e8(%rbp)
  403555:	c5 f8 28 85 30 ff ff 	vmovaps -0xd0(%rbp),%xmm0
  40355c:	ff 
  40355d:	c5 f8 29 85 00 fb ff 	vmovaps %xmm0,-0x500(%rbp)
  403564:	ff 
  403565:	48 8b 85 18 fb ff ff 	mov    -0x4e8(%rbp),%rax
  40356c:	c5 f8 28 85 00 fb ff 	vmovaps -0x500(%rbp),%xmm0
  403573:	ff 
  403574:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  403578:	8b 45 30             	mov    0x30(%rbp),%eax
  40357b:	48 63 d0             	movslq %eax,%rdx
  40357e:	8b 45 10             	mov    0x10(%rbp),%eax
  403581:	48 98                	cltq   
  403583:	48 0f af d0          	imul   %rax,%rdx
  403587:	8b 45 18             	mov    0x18(%rbp),%eax
  40358a:	48 98                	cltq   
  40358c:	48 01 d0             	add    %rdx,%rax
  40358f:	48 c1 e0 04          	shl    $0x4,%rax
  403593:	48 89 c2             	mov    %rax,%rdx
  403596:	48 8b 85 58 f2 ff ff 	mov    -0xda8(%rbp),%rax
  40359d:	48 01 d0             	add    %rdx,%rax
  4035a0:	48 89 85 f8 fa ff ff 	mov    %rax,-0x508(%rbp)
  4035a7:	c5 f8 28 85 20 ff ff 	vmovaps -0xe0(%rbp),%xmm0
  4035ae:	ff 
  4035af:	c5 f8 29 85 e0 fa ff 	vmovaps %xmm0,-0x520(%rbp)
  4035b6:	ff 
  4035b7:	48 8b 85 f8 fa ff ff 	mov    -0x508(%rbp),%rax
  4035be:	c5 f8 28 85 e0 fa ff 	vmovaps -0x520(%rbp),%xmm0
  4035c5:	ff 
  4035c6:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  4035ca:	8b 45 30             	mov    0x30(%rbp),%eax
  4035cd:	48 63 d0             	movslq %eax,%rdx
  4035d0:	8b 45 10             	mov    0x10(%rbp),%eax
  4035d3:	48 98                	cltq   
  4035d5:	48 0f af d0          	imul   %rax,%rdx
  4035d9:	8b 45 18             	mov    0x18(%rbp),%eax
  4035dc:	48 98                	cltq   
  4035de:	48 01 d0             	add    %rdx,%rax
  4035e1:	48 c1 e0 04          	shl    $0x4,%rax
  4035e5:	48 89 c2             	mov    %rax,%rdx
  4035e8:	48 8b 85 50 f2 ff ff 	mov    -0xdb0(%rbp),%rax
  4035ef:	48 01 d0             	add    %rdx,%rax
  4035f2:	48 89 85 d8 fa ff ff 	mov    %rax,-0x528(%rbp)
  4035f9:	c5 f8 28 85 10 ff ff 	vmovaps -0xf0(%rbp),%xmm0
  403600:	ff 
  403601:	c5 f8 29 85 c0 fa ff 	vmovaps %xmm0,-0x540(%rbp)
  403608:	ff 
  403609:	48 8b 85 d8 fa ff ff 	mov    -0x528(%rbp),%rax
  403610:	c5 f8 28 85 c0 fa ff 	vmovaps -0x540(%rbp),%xmm0
  403617:	ff 
  403618:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  40361c:	48 8b 45 20          	mov    0x20(%rbp),%rax
  403620:	48 83 c0 40          	add    $0x40,%rax
  403624:	48 89 85 b8 fa ff ff 	mov    %rax,-0x548(%rbp)
  40362b:	48 8b 85 b8 fa ff ff 	mov    -0x548(%rbp),%rax
  403632:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  403636:	c5 f8 29 85 70 fe ff 	vmovaps %xmm0,-0x190(%rbp)
  40363d:	ff 
  40363e:	48 8b 45 20          	mov    0x20(%rbp),%rax
  403642:	48 83 c0 50          	add    $0x50,%rax
  403646:	48 89 85 b0 fa ff ff 	mov    %rax,-0x550(%rbp)
  40364d:	48 8b 85 b0 fa ff ff 	mov    -0x550(%rbp),%rax
  403654:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  403658:	c5 f8 29 85 60 fe ff 	vmovaps %xmm0,-0x1a0(%rbp)
  40365f:	ff 
  403660:	48 8b 45 20          	mov    0x20(%rbp),%rax
  403664:	48 83 c0 60          	add    $0x60,%rax
  403668:	48 89 85 a8 fa ff ff 	mov    %rax,-0x558(%rbp)
  40366f:	48 8b 85 a8 fa ff ff 	mov    -0x558(%rbp),%rax
  403676:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  40367a:	c5 f8 29 85 50 fe ff 	vmovaps %xmm0,-0x1b0(%rbp)
  403681:	ff 
  403682:	48 8b 45 20          	mov    0x20(%rbp),%rax
  403686:	48 83 c0 70          	add    $0x70,%rax
  40368a:	48 89 85 a0 fa ff ff 	mov    %rax,-0x560(%rbp)
  403691:	48 8b 85 a0 fa ff ff 	mov    -0x560(%rbp),%rax
  403698:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  40369c:	c5 f8 29 85 40 fe ff 	vmovaps %xmm0,-0x1c0(%rbp)
  4036a3:	ff 
  4036a4:	c5 f8 28 45 f0       	vmovaps -0x10(%rbp),%xmm0
  4036a9:	c5 f8 29 85 90 fa ff 	vmovaps %xmm0,-0x570(%rbp)
  4036b0:	ff 
  4036b1:	c5 f8 28 85 70 fe ff 	vmovaps -0x190(%rbp),%xmm0
  4036b8:	ff 
  4036b9:	c5 f8 29 85 80 fa ff 	vmovaps %xmm0,-0x580(%rbp)
  4036c0:	ff 
  4036c1:	c5 f8 28 85 00 ff ff 	vmovaps -0x100(%rbp),%xmm0
  4036c8:	ff 
  4036c9:	c5 f8 29 85 70 fa ff 	vmovaps %xmm0,-0x590(%rbp)
  4036d0:	ff 
  4036d1:	c5 f8 28 8d 80 fa ff 	vmovaps -0x580(%rbp),%xmm1
  4036d8:	ff 
  4036d9:	c5 f8 28 85 70 fa ff 	vmovaps -0x590(%rbp),%xmm0
  4036e0:	ff 
  4036e1:	c4 e2 71 b8 85 90 fa 	vfmadd231ps -0x570(%rbp),%xmm1,%xmm0
  4036e8:	ff ff 
  4036ea:	c5 f8 29 85 00 ff ff 	vmovaps %xmm0,-0x100(%rbp)
  4036f1:	ff 
  4036f2:	c5 f8 28 45 e0       	vmovaps -0x20(%rbp),%xmm0
  4036f7:	c5 f8 29 85 60 fa ff 	vmovaps %xmm0,-0x5a0(%rbp)
  4036fe:	ff 
  4036ff:	c5 f8 28 85 60 fe ff 	vmovaps -0x1a0(%rbp),%xmm0
  403706:	ff 
  403707:	c5 f8 29 85 50 fa ff 	vmovaps %xmm0,-0x5b0(%rbp)
  40370e:	ff 
  40370f:	c5 f8 28 85 00 ff ff 	vmovaps -0x100(%rbp),%xmm0
  403716:	ff 
  403717:	c5 f8 29 85 40 fa ff 	vmovaps %xmm0,-0x5c0(%rbp)
  40371e:	ff 
  40371f:	c5 f8 28 8d 50 fa ff 	vmovaps -0x5b0(%rbp),%xmm1
  403726:	ff 
  403727:	c5 f8 28 85 40 fa ff 	vmovaps -0x5c0(%rbp),%xmm0
  40372e:	ff 
  40372f:	c4 e2 71 b8 85 60 fa 	vfmadd231ps -0x5a0(%rbp),%xmm1,%xmm0
  403736:	ff ff 
  403738:	c5 f8 29 85 00 ff ff 	vmovaps %xmm0,-0x100(%rbp)
  40373f:	ff 
  403740:	c5 f8 28 45 d0       	vmovaps -0x30(%rbp),%xmm0
  403745:	c5 f8 29 85 30 fa ff 	vmovaps %xmm0,-0x5d0(%rbp)
  40374c:	ff 
  40374d:	c5 f8 28 85 50 fe ff 	vmovaps -0x1b0(%rbp),%xmm0
  403754:	ff 
  403755:	c5 f8 29 85 20 fa ff 	vmovaps %xmm0,-0x5e0(%rbp)
  40375c:	ff 
  40375d:	c5 f8 28 85 00 ff ff 	vmovaps -0x100(%rbp),%xmm0
  403764:	ff 
  403765:	c5 f8 29 85 10 fa ff 	vmovaps %xmm0,-0x5f0(%rbp)
  40376c:	ff 
  40376d:	c5 f8 28 8d 20 fa ff 	vmovaps -0x5e0(%rbp),%xmm1
  403774:	ff 
  403775:	c5 f8 28 85 10 fa ff 	vmovaps -0x5f0(%rbp),%xmm0
  40377c:	ff 
  40377d:	c4 e2 71 b8 85 30 fa 	vfmadd231ps -0x5d0(%rbp),%xmm1,%xmm0
  403784:	ff ff 
  403786:	c5 f8 29 85 00 ff ff 	vmovaps %xmm0,-0x100(%rbp)
  40378d:	ff 
  40378e:	c5 f8 28 45 c0       	vmovaps -0x40(%rbp),%xmm0
  403793:	c5 f8 29 85 00 fa ff 	vmovaps %xmm0,-0x600(%rbp)
  40379a:	ff 
  40379b:	c5 f8 28 85 40 fe ff 	vmovaps -0x1c0(%rbp),%xmm0
  4037a2:	ff 
  4037a3:	c5 f8 29 85 f0 f9 ff 	vmovaps %xmm0,-0x610(%rbp)
  4037aa:	ff 
  4037ab:	c5 f8 28 85 00 ff ff 	vmovaps -0x100(%rbp),%xmm0
  4037b2:	ff 
  4037b3:	c5 f8 29 85 e0 f9 ff 	vmovaps %xmm0,-0x620(%rbp)
  4037ba:	ff 
  4037bb:	c5 f8 28 8d f0 f9 ff 	vmovaps -0x610(%rbp),%xmm1
  4037c2:	ff 
  4037c3:	c5 f8 28 85 e0 f9 ff 	vmovaps -0x620(%rbp),%xmm0
  4037ca:	ff 
  4037cb:	c4 e2 71 b8 85 00 fa 	vfmadd231ps -0x600(%rbp),%xmm1,%xmm0
  4037d2:	ff ff 
  4037d4:	c5 f8 29 85 00 ff ff 	vmovaps %xmm0,-0x100(%rbp)
  4037db:	ff 
  4037dc:	c5 f8 28 45 b0       	vmovaps -0x50(%rbp),%xmm0
  4037e1:	c5 f8 29 85 d0 f9 ff 	vmovaps %xmm0,-0x630(%rbp)
  4037e8:	ff 
  4037e9:	c5 f8 28 85 70 fe ff 	vmovaps -0x190(%rbp),%xmm0
  4037f0:	ff 
  4037f1:	c5 f8 29 85 c0 f9 ff 	vmovaps %xmm0,-0x640(%rbp)
  4037f8:	ff 
  4037f9:	c5 f8 28 85 f0 fe ff 	vmovaps -0x110(%rbp),%xmm0
  403800:	ff 
  403801:	c5 f8 29 85 b0 f9 ff 	vmovaps %xmm0,-0x650(%rbp)
  403808:	ff 
  403809:	c5 f8 28 8d c0 f9 ff 	vmovaps -0x640(%rbp),%xmm1
  403810:	ff 
  403811:	c5 f8 28 85 b0 f9 ff 	vmovaps -0x650(%rbp),%xmm0
  403818:	ff 
  403819:	c4 e2 71 b8 85 d0 f9 	vfmadd231ps -0x630(%rbp),%xmm1,%xmm0
  403820:	ff ff 
  403822:	c5 f8 29 85 f0 fe ff 	vmovaps %xmm0,-0x110(%rbp)
  403829:	ff 
  40382a:	c5 f8 28 45 a0       	vmovaps -0x60(%rbp),%xmm0
  40382f:	c5 f8 29 85 a0 f9 ff 	vmovaps %xmm0,-0x660(%rbp)
  403836:	ff 
  403837:	c5 f8 28 85 60 fe ff 	vmovaps -0x1a0(%rbp),%xmm0
  40383e:	ff 
  40383f:	c5 f8 29 85 90 f9 ff 	vmovaps %xmm0,-0x670(%rbp)
  403846:	ff 
  403847:	c5 f8 28 85 f0 fe ff 	vmovaps -0x110(%rbp),%xmm0
  40384e:	ff 
  40384f:	c5 f8 29 85 80 f9 ff 	vmovaps %xmm0,-0x680(%rbp)
  403856:	ff 
  403857:	c5 f8 28 8d 90 f9 ff 	vmovaps -0x670(%rbp),%xmm1
  40385e:	ff 
  40385f:	c5 f8 28 85 80 f9 ff 	vmovaps -0x680(%rbp),%xmm0
  403866:	ff 
  403867:	c4 e2 71 b8 85 a0 f9 	vfmadd231ps -0x660(%rbp),%xmm1,%xmm0
  40386e:	ff ff 
  403870:	c5 f8 29 85 f0 fe ff 	vmovaps %xmm0,-0x110(%rbp)
  403877:	ff 
  403878:	c5 f8 28 45 90       	vmovaps -0x70(%rbp),%xmm0
  40387d:	c5 f8 29 85 70 f9 ff 	vmovaps %xmm0,-0x690(%rbp)
  403884:	ff 
  403885:	c5 f8 28 85 50 fe ff 	vmovaps -0x1b0(%rbp),%xmm0
  40388c:	ff 
  40388d:	c5 f8 29 85 60 f9 ff 	vmovaps %xmm0,-0x6a0(%rbp)
  403894:	ff 
  403895:	c5 f8 28 85 f0 fe ff 	vmovaps -0x110(%rbp),%xmm0
  40389c:	ff 
  40389d:	c5 f8 29 85 50 f9 ff 	vmovaps %xmm0,-0x6b0(%rbp)
  4038a4:	ff 
  4038a5:	c5 f8 28 8d 60 f9 ff 	vmovaps -0x6a0(%rbp),%xmm1
  4038ac:	ff 
  4038ad:	c5 f8 28 85 50 f9 ff 	vmovaps -0x6b0(%rbp),%xmm0
  4038b4:	ff 
  4038b5:	c4 e2 71 b8 85 70 f9 	vfmadd231ps -0x690(%rbp),%xmm1,%xmm0
  4038bc:	ff ff 
  4038be:	c5 f8 29 85 f0 fe ff 	vmovaps %xmm0,-0x110(%rbp)
  4038c5:	ff 
  4038c6:	c5 f8 28 45 80       	vmovaps -0x80(%rbp),%xmm0
  4038cb:	c5 f8 29 85 40 f9 ff 	vmovaps %xmm0,-0x6c0(%rbp)
  4038d2:	ff 
  4038d3:	c5 f8 28 85 40 fe ff 	vmovaps -0x1c0(%rbp),%xmm0
  4038da:	ff 
  4038db:	c5 f8 29 85 30 f9 ff 	vmovaps %xmm0,-0x6d0(%rbp)
  4038e2:	ff 
  4038e3:	c5 f8 28 85 f0 fe ff 	vmovaps -0x110(%rbp),%xmm0
  4038ea:	ff 
  4038eb:	c5 f8 29 85 20 f9 ff 	vmovaps %xmm0,-0x6e0(%rbp)
  4038f2:	ff 
  4038f3:	c5 f8 28 8d 30 f9 ff 	vmovaps -0x6d0(%rbp),%xmm1
  4038fa:	ff 
  4038fb:	c5 f8 28 85 20 f9 ff 	vmovaps -0x6e0(%rbp),%xmm0
  403902:	ff 
  403903:	c4 e2 71 b8 85 40 f9 	vfmadd231ps -0x6c0(%rbp),%xmm1,%xmm0
  40390a:	ff ff 
  40390c:	c5 f8 29 85 f0 fe ff 	vmovaps %xmm0,-0x110(%rbp)
  403913:	ff 
  403914:	c5 f8 28 85 70 ff ff 	vmovaps -0x90(%rbp),%xmm0
  40391b:	ff 
  40391c:	c5 f8 29 85 10 f9 ff 	vmovaps %xmm0,-0x6f0(%rbp)
  403923:	ff 
  403924:	c5 f8 28 85 70 fe ff 	vmovaps -0x190(%rbp),%xmm0
  40392b:	ff 
  40392c:	c5 f8 29 85 00 f9 ff 	vmovaps %xmm0,-0x700(%rbp)
  403933:	ff 
  403934:	c5 f8 28 85 e0 fe ff 	vmovaps -0x120(%rbp),%xmm0
  40393b:	ff 
  40393c:	c5 f8 29 85 f0 f8 ff 	vmovaps %xmm0,-0x710(%rbp)
  403943:	ff 
  403944:	c5 f8 28 8d 00 f9 ff 	vmovaps -0x700(%rbp),%xmm1
  40394b:	ff 
  40394c:	c5 f8 28 85 f0 f8 ff 	vmovaps -0x710(%rbp),%xmm0
  403953:	ff 
  403954:	c4 e2 71 b8 85 10 f9 	vfmadd231ps -0x6f0(%rbp),%xmm1,%xmm0
  40395b:	ff ff 
  40395d:	c5 f8 29 85 e0 fe ff 	vmovaps %xmm0,-0x120(%rbp)
  403964:	ff 
  403965:	c5 f8 28 85 60 ff ff 	vmovaps -0xa0(%rbp),%xmm0
  40396c:	ff 
  40396d:	c5 f8 29 85 e0 f8 ff 	vmovaps %xmm0,-0x720(%rbp)
  403974:	ff 
  403975:	c5 f8 28 85 60 fe ff 	vmovaps -0x1a0(%rbp),%xmm0
  40397c:	ff 
  40397d:	c5 f8 29 85 d0 f8 ff 	vmovaps %xmm0,-0x730(%rbp)
  403984:	ff 
  403985:	c5 f8 28 85 e0 fe ff 	vmovaps -0x120(%rbp),%xmm0
  40398c:	ff 
  40398d:	c5 f8 29 85 c0 f8 ff 	vmovaps %xmm0,-0x740(%rbp)
  403994:	ff 
  403995:	c5 f8 28 8d d0 f8 ff 	vmovaps -0x730(%rbp),%xmm1
  40399c:	ff 
  40399d:	c5 f8 28 85 c0 f8 ff 	vmovaps -0x740(%rbp),%xmm0
  4039a4:	ff 
  4039a5:	c4 e2 71 b8 85 e0 f8 	vfmadd231ps -0x720(%rbp),%xmm1,%xmm0
  4039ac:	ff ff 
  4039ae:	c5 f8 29 85 e0 fe ff 	vmovaps %xmm0,-0x120(%rbp)
  4039b5:	ff 
  4039b6:	c5 f8 28 85 50 ff ff 	vmovaps -0xb0(%rbp),%xmm0
  4039bd:	ff 
  4039be:	c5 f8 29 85 b0 f8 ff 	vmovaps %xmm0,-0x750(%rbp)
  4039c5:	ff 
  4039c6:	c5 f8 28 85 50 fe ff 	vmovaps -0x1b0(%rbp),%xmm0
  4039cd:	ff 
  4039ce:	c5 f8 29 85 a0 f8 ff 	vmovaps %xmm0,-0x760(%rbp)
  4039d5:	ff 
  4039d6:	c5 f8 28 85 e0 fe ff 	vmovaps -0x120(%rbp),%xmm0
  4039dd:	ff 
  4039de:	c5 f8 29 85 90 f8 ff 	vmovaps %xmm0,-0x770(%rbp)
  4039e5:	ff 
  4039e6:	c5 f8 28 8d a0 f8 ff 	vmovaps -0x760(%rbp),%xmm1
  4039ed:	ff 
  4039ee:	c5 f8 28 85 90 f8 ff 	vmovaps -0x770(%rbp),%xmm0
  4039f5:	ff 
  4039f6:	c4 e2 71 b8 85 b0 f8 	vfmadd231ps -0x750(%rbp),%xmm1,%xmm0
  4039fd:	ff ff 
  4039ff:	c5 f8 29 85 e0 fe ff 	vmovaps %xmm0,-0x120(%rbp)
  403a06:	ff 
  403a07:	c5 f8 28 85 40 ff ff 	vmovaps -0xc0(%rbp),%xmm0
  403a0e:	ff 
  403a0f:	c5 f8 29 85 80 f8 ff 	vmovaps %xmm0,-0x780(%rbp)
  403a16:	ff 
  403a17:	c5 f8 28 85 40 fe ff 	vmovaps -0x1c0(%rbp),%xmm0
  403a1e:	ff 
  403a1f:	c5 f8 29 85 70 f8 ff 	vmovaps %xmm0,-0x790(%rbp)
  403a26:	ff 
  403a27:	c5 f8 28 85 e0 fe ff 	vmovaps -0x120(%rbp),%xmm0
  403a2e:	ff 
  403a2f:	c5 f8 29 85 60 f8 ff 	vmovaps %xmm0,-0x7a0(%rbp)
  403a36:	ff 
  403a37:	c5 f8 28 8d 70 f8 ff 	vmovaps -0x790(%rbp),%xmm1
  403a3e:	ff 
  403a3f:	c5 f8 28 85 60 f8 ff 	vmovaps -0x7a0(%rbp),%xmm0
  403a46:	ff 
  403a47:	c4 e2 71 b8 85 80 f8 	vfmadd231ps -0x780(%rbp),%xmm1,%xmm0
  403a4e:	ff ff 
  403a50:	c5 f8 29 85 e0 fe ff 	vmovaps %xmm0,-0x120(%rbp)
  403a57:	ff 
  403a58:	8b 45 30             	mov    0x30(%rbp),%eax
  403a5b:	c1 e0 02             	shl    $0x2,%eax
  403a5e:	0f af 45 10          	imul   0x10(%rbp),%eax
  403a62:	8b 55 18             	mov    0x18(%rbp),%edx
  403a65:	c1 e2 02             	shl    $0x2,%edx
  403a68:	01 c2                	add    %eax,%edx
  403a6a:	8b 45 30             	mov    0x30(%rbp),%eax
  403a6d:	01 d0                	add    %edx,%eax
  403a6f:	48 98                	cltq   
  403a71:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  403a78:	00 
  403a79:	48 8b 85 60 f2 ff ff 	mov    -0xda0(%rbp),%rax
  403a80:	48 01 d0             	add    %rdx,%rax
  403a83:	48 89 85 58 f8 ff ff 	mov    %rax,-0x7a8(%rbp)
  403a8a:	c5 f8 28 85 00 ff ff 	vmovaps -0x100(%rbp),%xmm0
  403a91:	ff 
  403a92:	c5 f8 29 85 40 f8 ff 	vmovaps %xmm0,-0x7c0(%rbp)
  403a99:	ff 
  403a9a:	48 8b 85 58 f8 ff ff 	mov    -0x7a8(%rbp),%rax
  403aa1:	c5 f8 28 85 40 f8 ff 	vmovaps -0x7c0(%rbp),%xmm0
  403aa8:	ff 
  403aa9:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  403aad:	8b 45 30             	mov    0x30(%rbp),%eax
  403ab0:	c1 e0 02             	shl    $0x2,%eax
  403ab3:	0f af 45 10          	imul   0x10(%rbp),%eax
  403ab7:	8b 55 18             	mov    0x18(%rbp),%edx
  403aba:	c1 e2 02             	shl    $0x2,%edx
  403abd:	01 c2                	add    %eax,%edx
  403abf:	8b 45 30             	mov    0x30(%rbp),%eax
  403ac2:	01 d0                	add    %edx,%eax
  403ac4:	48 98                	cltq   
  403ac6:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  403acd:	00 
  403ace:	48 8b 85 58 f2 ff ff 	mov    -0xda8(%rbp),%rax
  403ad5:	48 01 d0             	add    %rdx,%rax
  403ad8:	48 89 85 38 f8 ff ff 	mov    %rax,-0x7c8(%rbp)
  403adf:	c5 f8 28 85 f0 fe ff 	vmovaps -0x110(%rbp),%xmm0
  403ae6:	ff 
  403ae7:	c5 f8 29 85 20 f8 ff 	vmovaps %xmm0,-0x7e0(%rbp)
  403aee:	ff 
  403aef:	48 8b 85 38 f8 ff ff 	mov    -0x7c8(%rbp),%rax
  403af6:	c5 f8 28 85 20 f8 ff 	vmovaps -0x7e0(%rbp),%xmm0
  403afd:	ff 
  403afe:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  403b02:	8b 45 30             	mov    0x30(%rbp),%eax
  403b05:	c1 e0 02             	shl    $0x2,%eax
  403b08:	0f af 45 10          	imul   0x10(%rbp),%eax
  403b0c:	8b 55 18             	mov    0x18(%rbp),%edx
  403b0f:	c1 e2 02             	shl    $0x2,%edx
  403b12:	01 c2                	add    %eax,%edx
  403b14:	8b 45 30             	mov    0x30(%rbp),%eax
  403b17:	01 d0                	add    %edx,%eax
  403b19:	48 98                	cltq   
  403b1b:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  403b22:	00 
  403b23:	48 8b 85 50 f2 ff ff 	mov    -0xdb0(%rbp),%rax
  403b2a:	48 01 d0             	add    %rdx,%rax
  403b2d:	48 89 85 18 f8 ff ff 	mov    %rax,-0x7e8(%rbp)
  403b34:	c5 f8 28 85 e0 fe ff 	vmovaps -0x120(%rbp),%xmm0
  403b3b:	ff 
  403b3c:	c5 f8 29 85 00 f8 ff 	vmovaps %xmm0,-0x800(%rbp)
  403b43:	ff 
  403b44:	48 8b 85 18 f8 ff ff 	mov    -0x7e8(%rbp),%rax
  403b4b:	c5 f8 28 85 00 f8 ff 	vmovaps -0x800(%rbp),%xmm0
  403b52:	ff 
  403b53:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  403b57:	48 8b 45 20          	mov    0x20(%rbp),%rax
  403b5b:	48 83 e8 80          	sub    $0xffffffffffffff80,%rax
  403b5f:	48 89 85 f8 f7 ff ff 	mov    %rax,-0x808(%rbp)
  403b66:	48 8b 85 f8 f7 ff ff 	mov    -0x808(%rbp),%rax
  403b6d:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  403b71:	c5 f8 29 85 70 fe ff 	vmovaps %xmm0,-0x190(%rbp)
  403b78:	ff 
  403b79:	48 8b 45 20          	mov    0x20(%rbp),%rax
  403b7d:	48 05 90 00 00 00    	add    $0x90,%rax
  403b83:	48 89 85 f0 f7 ff ff 	mov    %rax,-0x810(%rbp)
  403b8a:	48 8b 85 f0 f7 ff ff 	mov    -0x810(%rbp),%rax
  403b91:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  403b95:	c5 f8 29 85 60 fe ff 	vmovaps %xmm0,-0x1a0(%rbp)
  403b9c:	ff 
  403b9d:	48 8b 45 20          	mov    0x20(%rbp),%rax
  403ba1:	48 05 a0 00 00 00    	add    $0xa0,%rax
  403ba7:	48 89 85 e8 f7 ff ff 	mov    %rax,-0x818(%rbp)
  403bae:	48 8b 85 e8 f7 ff ff 	mov    -0x818(%rbp),%rax
  403bb5:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  403bb9:	c5 f8 29 85 50 fe ff 	vmovaps %xmm0,-0x1b0(%rbp)
  403bc0:	ff 
  403bc1:	48 8b 45 20          	mov    0x20(%rbp),%rax
  403bc5:	48 05 b0 00 00 00    	add    $0xb0,%rax
  403bcb:	48 89 85 e0 f7 ff ff 	mov    %rax,-0x820(%rbp)
  403bd2:	48 8b 85 e0 f7 ff ff 	mov    -0x820(%rbp),%rax
  403bd9:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  403bdd:	c5 f8 29 85 40 fe ff 	vmovaps %xmm0,-0x1c0(%rbp)
  403be4:	ff 
  403be5:	c5 f8 28 45 f0       	vmovaps -0x10(%rbp),%xmm0
  403bea:	c5 f8 29 85 d0 f7 ff 	vmovaps %xmm0,-0x830(%rbp)
  403bf1:	ff 
  403bf2:	c5 f8 28 85 70 fe ff 	vmovaps -0x190(%rbp),%xmm0
  403bf9:	ff 
  403bfa:	c5 f8 29 85 c0 f7 ff 	vmovaps %xmm0,-0x840(%rbp)
  403c01:	ff 
  403c02:	c5 f8 28 85 d0 fe ff 	vmovaps -0x130(%rbp),%xmm0
  403c09:	ff 
  403c0a:	c5 f8 29 85 b0 f7 ff 	vmovaps %xmm0,-0x850(%rbp)
  403c11:	ff 
  403c12:	c5 f8 28 8d c0 f7 ff 	vmovaps -0x840(%rbp),%xmm1
  403c19:	ff 
  403c1a:	c5 f8 28 85 b0 f7 ff 	vmovaps -0x850(%rbp),%xmm0
  403c21:	ff 
  403c22:	c4 e2 71 b8 85 d0 f7 	vfmadd231ps -0x830(%rbp),%xmm1,%xmm0
  403c29:	ff ff 
  403c2b:	c5 f8 29 85 d0 fe ff 	vmovaps %xmm0,-0x130(%rbp)
  403c32:	ff 
  403c33:	c5 f8 28 45 e0       	vmovaps -0x20(%rbp),%xmm0
  403c38:	c5 f8 29 85 a0 f7 ff 	vmovaps %xmm0,-0x860(%rbp)
  403c3f:	ff 
  403c40:	c5 f8 28 85 60 fe ff 	vmovaps -0x1a0(%rbp),%xmm0
  403c47:	ff 
  403c48:	c5 f8 29 85 90 f7 ff 	vmovaps %xmm0,-0x870(%rbp)
  403c4f:	ff 
  403c50:	c5 f8 28 85 d0 fe ff 	vmovaps -0x130(%rbp),%xmm0
  403c57:	ff 
  403c58:	c5 f8 29 85 80 f7 ff 	vmovaps %xmm0,-0x880(%rbp)
  403c5f:	ff 
  403c60:	c5 f8 28 8d 90 f7 ff 	vmovaps -0x870(%rbp),%xmm1
  403c67:	ff 
  403c68:	c5 f8 28 85 80 f7 ff 	vmovaps -0x880(%rbp),%xmm0
  403c6f:	ff 
  403c70:	c4 e2 71 b8 85 a0 f7 	vfmadd231ps -0x860(%rbp),%xmm1,%xmm0
  403c77:	ff ff 
  403c79:	c5 f8 29 85 d0 fe ff 	vmovaps %xmm0,-0x130(%rbp)
  403c80:	ff 
  403c81:	c5 f8 28 45 d0       	vmovaps -0x30(%rbp),%xmm0
  403c86:	c5 f8 29 85 70 f7 ff 	vmovaps %xmm0,-0x890(%rbp)
  403c8d:	ff 
  403c8e:	c5 f8 28 85 50 fe ff 	vmovaps -0x1b0(%rbp),%xmm0
  403c95:	ff 
  403c96:	c5 f8 29 85 60 f7 ff 	vmovaps %xmm0,-0x8a0(%rbp)
  403c9d:	ff 
  403c9e:	c5 f8 28 85 d0 fe ff 	vmovaps -0x130(%rbp),%xmm0
  403ca5:	ff 
  403ca6:	c5 f8 29 85 50 f7 ff 	vmovaps %xmm0,-0x8b0(%rbp)
  403cad:	ff 
  403cae:	c5 f8 28 8d 60 f7 ff 	vmovaps -0x8a0(%rbp),%xmm1
  403cb5:	ff 
  403cb6:	c5 f8 28 85 50 f7 ff 	vmovaps -0x8b0(%rbp),%xmm0
  403cbd:	ff 
  403cbe:	c4 e2 71 b8 85 70 f7 	vfmadd231ps -0x890(%rbp),%xmm1,%xmm0
  403cc5:	ff ff 
  403cc7:	c5 f8 29 85 d0 fe ff 	vmovaps %xmm0,-0x130(%rbp)
  403cce:	ff 
  403ccf:	c5 f8 28 45 c0       	vmovaps -0x40(%rbp),%xmm0
  403cd4:	c5 f8 29 85 40 f7 ff 	vmovaps %xmm0,-0x8c0(%rbp)
  403cdb:	ff 
  403cdc:	c5 f8 28 85 40 fe ff 	vmovaps -0x1c0(%rbp),%xmm0
  403ce3:	ff 
  403ce4:	c5 f8 29 85 30 f7 ff 	vmovaps %xmm0,-0x8d0(%rbp)
  403ceb:	ff 
  403cec:	c5 f8 28 85 d0 fe ff 	vmovaps -0x130(%rbp),%xmm0
  403cf3:	ff 
  403cf4:	c5 f8 29 85 20 f7 ff 	vmovaps %xmm0,-0x8e0(%rbp)
  403cfb:	ff 
  403cfc:	c5 f8 28 8d 30 f7 ff 	vmovaps -0x8d0(%rbp),%xmm1
  403d03:	ff 
  403d04:	c5 f8 28 85 20 f7 ff 	vmovaps -0x8e0(%rbp),%xmm0
  403d0b:	ff 
  403d0c:	c4 e2 71 b8 85 40 f7 	vfmadd231ps -0x8c0(%rbp),%xmm1,%xmm0
  403d13:	ff ff 
  403d15:	c5 f8 29 85 d0 fe ff 	vmovaps %xmm0,-0x130(%rbp)
  403d1c:	ff 
  403d1d:	c5 f8 28 45 b0       	vmovaps -0x50(%rbp),%xmm0
  403d22:	c5 f8 29 85 10 f7 ff 	vmovaps %xmm0,-0x8f0(%rbp)
  403d29:	ff 
  403d2a:	c5 f8 28 85 70 fe ff 	vmovaps -0x190(%rbp),%xmm0
  403d31:	ff 
  403d32:	c5 f8 29 85 00 f7 ff 	vmovaps %xmm0,-0x900(%rbp)
  403d39:	ff 
  403d3a:	c5 f8 28 85 c0 fe ff 	vmovaps -0x140(%rbp),%xmm0
  403d41:	ff 
  403d42:	c5 f8 29 85 f0 f6 ff 	vmovaps %xmm0,-0x910(%rbp)
  403d49:	ff 
  403d4a:	c5 f8 28 8d 00 f7 ff 	vmovaps -0x900(%rbp),%xmm1
  403d51:	ff 
  403d52:	c5 f8 28 85 f0 f6 ff 	vmovaps -0x910(%rbp),%xmm0
  403d59:	ff 
  403d5a:	c4 e2 71 b8 85 10 f7 	vfmadd231ps -0x8f0(%rbp),%xmm1,%xmm0
  403d61:	ff ff 
  403d63:	c5 f8 29 85 c0 fe ff 	vmovaps %xmm0,-0x140(%rbp)
  403d6a:	ff 
  403d6b:	c5 f8 28 45 a0       	vmovaps -0x60(%rbp),%xmm0
  403d70:	c5 f8 29 85 e0 f6 ff 	vmovaps %xmm0,-0x920(%rbp)
  403d77:	ff 
  403d78:	c5 f8 28 85 60 fe ff 	vmovaps -0x1a0(%rbp),%xmm0
  403d7f:	ff 
  403d80:	c5 f8 29 85 d0 f6 ff 	vmovaps %xmm0,-0x930(%rbp)
  403d87:	ff 
  403d88:	c5 f8 28 85 c0 fe ff 	vmovaps -0x140(%rbp),%xmm0
  403d8f:	ff 
  403d90:	c5 f8 29 85 c0 f6 ff 	vmovaps %xmm0,-0x940(%rbp)
  403d97:	ff 
  403d98:	c5 f8 28 8d d0 f6 ff 	vmovaps -0x930(%rbp),%xmm1
  403d9f:	ff 
  403da0:	c5 f8 28 85 c0 f6 ff 	vmovaps -0x940(%rbp),%xmm0
  403da7:	ff 
  403da8:	c4 e2 71 b8 85 e0 f6 	vfmadd231ps -0x920(%rbp),%xmm1,%xmm0
  403daf:	ff ff 
  403db1:	c5 f8 29 85 c0 fe ff 	vmovaps %xmm0,-0x140(%rbp)
  403db8:	ff 
  403db9:	c5 f8 28 45 90       	vmovaps -0x70(%rbp),%xmm0
  403dbe:	c5 f8 29 85 b0 f6 ff 	vmovaps %xmm0,-0x950(%rbp)
  403dc5:	ff 
  403dc6:	c5 f8 28 85 50 fe ff 	vmovaps -0x1b0(%rbp),%xmm0
  403dcd:	ff 
  403dce:	c5 f8 29 85 a0 f6 ff 	vmovaps %xmm0,-0x960(%rbp)
  403dd5:	ff 
  403dd6:	c5 f8 28 85 c0 fe ff 	vmovaps -0x140(%rbp),%xmm0
  403ddd:	ff 
  403dde:	c5 f8 29 85 90 f6 ff 	vmovaps %xmm0,-0x970(%rbp)
  403de5:	ff 
  403de6:	c5 f8 28 8d a0 f6 ff 	vmovaps -0x960(%rbp),%xmm1
  403ded:	ff 
  403dee:	c5 f8 28 85 90 f6 ff 	vmovaps -0x970(%rbp),%xmm0
  403df5:	ff 
  403df6:	c4 e2 71 b8 85 b0 f6 	vfmadd231ps -0x950(%rbp),%xmm1,%xmm0
  403dfd:	ff ff 
  403dff:	c5 f8 29 85 c0 fe ff 	vmovaps %xmm0,-0x140(%rbp)
  403e06:	ff 
  403e07:	c5 f8 28 45 80       	vmovaps -0x80(%rbp),%xmm0
  403e0c:	c5 f8 29 85 80 f6 ff 	vmovaps %xmm0,-0x980(%rbp)
  403e13:	ff 
  403e14:	c5 f8 28 85 40 fe ff 	vmovaps -0x1c0(%rbp),%xmm0
  403e1b:	ff 
  403e1c:	c5 f8 29 85 70 f6 ff 	vmovaps %xmm0,-0x990(%rbp)
  403e23:	ff 
  403e24:	c5 f8 28 85 c0 fe ff 	vmovaps -0x140(%rbp),%xmm0
  403e2b:	ff 
  403e2c:	c5 f8 29 85 60 f6 ff 	vmovaps %xmm0,-0x9a0(%rbp)
  403e33:	ff 
  403e34:	c5 f8 28 8d 70 f6 ff 	vmovaps -0x990(%rbp),%xmm1
  403e3b:	ff 
  403e3c:	c5 f8 28 85 60 f6 ff 	vmovaps -0x9a0(%rbp),%xmm0
  403e43:	ff 
  403e44:	c4 e2 71 b8 85 80 f6 	vfmadd231ps -0x980(%rbp),%xmm1,%xmm0
  403e4b:	ff ff 
  403e4d:	c5 f8 29 85 c0 fe ff 	vmovaps %xmm0,-0x140(%rbp)
  403e54:	ff 
  403e55:	c5 f8 28 85 70 ff ff 	vmovaps -0x90(%rbp),%xmm0
  403e5c:	ff 
  403e5d:	c5 f8 29 85 50 f6 ff 	vmovaps %xmm0,-0x9b0(%rbp)
  403e64:	ff 
  403e65:	c5 f8 28 85 70 fe ff 	vmovaps -0x190(%rbp),%xmm0
  403e6c:	ff 
  403e6d:	c5 f8 29 85 40 f6 ff 	vmovaps %xmm0,-0x9c0(%rbp)
  403e74:	ff 
  403e75:	c5 f8 28 85 b0 fe ff 	vmovaps -0x150(%rbp),%xmm0
  403e7c:	ff 
  403e7d:	c5 f8 29 85 30 f6 ff 	vmovaps %xmm0,-0x9d0(%rbp)
  403e84:	ff 
  403e85:	c5 f8 28 8d 40 f6 ff 	vmovaps -0x9c0(%rbp),%xmm1
  403e8c:	ff 
  403e8d:	c5 f8 28 85 30 f6 ff 	vmovaps -0x9d0(%rbp),%xmm0
  403e94:	ff 
  403e95:	c4 e2 71 b8 85 50 f6 	vfmadd231ps -0x9b0(%rbp),%xmm1,%xmm0
  403e9c:	ff ff 
  403e9e:	c5 f8 29 85 b0 fe ff 	vmovaps %xmm0,-0x150(%rbp)
  403ea5:	ff 
  403ea6:	c5 f8 28 85 60 ff ff 	vmovaps -0xa0(%rbp),%xmm0
  403ead:	ff 
  403eae:	c5 f8 29 85 20 f6 ff 	vmovaps %xmm0,-0x9e0(%rbp)
  403eb5:	ff 
  403eb6:	c5 f8 28 85 60 fe ff 	vmovaps -0x1a0(%rbp),%xmm0
  403ebd:	ff 
  403ebe:	c5 f8 29 85 10 f6 ff 	vmovaps %xmm0,-0x9f0(%rbp)
  403ec5:	ff 
  403ec6:	c5 f8 28 85 b0 fe ff 	vmovaps -0x150(%rbp),%xmm0
  403ecd:	ff 
  403ece:	c5 f8 29 85 00 f6 ff 	vmovaps %xmm0,-0xa00(%rbp)
  403ed5:	ff 
  403ed6:	c5 f8 28 8d 10 f6 ff 	vmovaps -0x9f0(%rbp),%xmm1
  403edd:	ff 
  403ede:	c5 f8 28 85 00 f6 ff 	vmovaps -0xa00(%rbp),%xmm0
  403ee5:	ff 
  403ee6:	c4 e2 71 b8 85 20 f6 	vfmadd231ps -0x9e0(%rbp),%xmm1,%xmm0
  403eed:	ff ff 
  403eef:	c5 f8 29 85 b0 fe ff 	vmovaps %xmm0,-0x150(%rbp)
  403ef6:	ff 
  403ef7:	c5 f8 28 85 50 ff ff 	vmovaps -0xb0(%rbp),%xmm0
  403efe:	ff 
  403eff:	c5 f8 29 85 f0 f5 ff 	vmovaps %xmm0,-0xa10(%rbp)
  403f06:	ff 
  403f07:	c5 f8 28 85 50 fe ff 	vmovaps -0x1b0(%rbp),%xmm0
  403f0e:	ff 
  403f0f:	c5 f8 29 85 e0 f5 ff 	vmovaps %xmm0,-0xa20(%rbp)
  403f16:	ff 
  403f17:	c5 f8 28 85 b0 fe ff 	vmovaps -0x150(%rbp),%xmm0
  403f1e:	ff 
  403f1f:	c5 f8 29 85 d0 f5 ff 	vmovaps %xmm0,-0xa30(%rbp)
  403f26:	ff 
  403f27:	c5 f8 28 8d e0 f5 ff 	vmovaps -0xa20(%rbp),%xmm1
  403f2e:	ff 
  403f2f:	c5 f8 28 85 d0 f5 ff 	vmovaps -0xa30(%rbp),%xmm0
  403f36:	ff 
  403f37:	c4 e2 71 b8 85 f0 f5 	vfmadd231ps -0xa10(%rbp),%xmm1,%xmm0
  403f3e:	ff ff 
  403f40:	c5 f8 29 85 b0 fe ff 	vmovaps %xmm0,-0x150(%rbp)
  403f47:	ff 
  403f48:	c5 f8 28 85 40 ff ff 	vmovaps -0xc0(%rbp),%xmm0
  403f4f:	ff 
  403f50:	c5 f8 29 85 c0 f5 ff 	vmovaps %xmm0,-0xa40(%rbp)
  403f57:	ff 
  403f58:	c5 f8 28 85 40 fe ff 	vmovaps -0x1c0(%rbp),%xmm0
  403f5f:	ff 
  403f60:	c5 f8 29 85 b0 f5 ff 	vmovaps %xmm0,-0xa50(%rbp)
  403f67:	ff 
  403f68:	c5 f8 28 85 b0 fe ff 	vmovaps -0x150(%rbp),%xmm0
  403f6f:	ff 
  403f70:	c5 f8 29 85 a0 f5 ff 	vmovaps %xmm0,-0xa60(%rbp)
  403f77:	ff 
  403f78:	c5 f8 28 8d b0 f5 ff 	vmovaps -0xa50(%rbp),%xmm1
  403f7f:	ff 
  403f80:	c5 f8 28 85 a0 f5 ff 	vmovaps -0xa60(%rbp),%xmm0
  403f87:	ff 
  403f88:	c4 e2 71 b8 85 c0 f5 	vfmadd231ps -0xa40(%rbp),%xmm1,%xmm0
  403f8f:	ff ff 
  403f91:	c5 f8 29 85 b0 fe ff 	vmovaps %xmm0,-0x150(%rbp)
  403f98:	ff 
  403f99:	8b 45 30             	mov    0x30(%rbp),%eax
  403f9c:	c1 e0 02             	shl    $0x2,%eax
  403f9f:	0f af 45 10          	imul   0x10(%rbp),%eax
  403fa3:	8b 55 18             	mov    0x18(%rbp),%edx
  403fa6:	c1 e2 02             	shl    $0x2,%edx
  403fa9:	01 c2                	add    %eax,%edx
  403fab:	8b 45 30             	mov    0x30(%rbp),%eax
  403fae:	01 c0                	add    %eax,%eax
  403fb0:	01 d0                	add    %edx,%eax
  403fb2:	48 98                	cltq   
  403fb4:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  403fbb:	00 
  403fbc:	48 8b 85 60 f2 ff ff 	mov    -0xda0(%rbp),%rax
  403fc3:	48 01 d0             	add    %rdx,%rax
  403fc6:	48 89 85 98 f5 ff ff 	mov    %rax,-0xa68(%rbp)
  403fcd:	c5 f8 28 85 d0 fe ff 	vmovaps -0x130(%rbp),%xmm0
  403fd4:	ff 
  403fd5:	c5 f8 29 85 80 f5 ff 	vmovaps %xmm0,-0xa80(%rbp)
  403fdc:	ff 
  403fdd:	48 8b 85 98 f5 ff ff 	mov    -0xa68(%rbp),%rax
  403fe4:	c5 f8 28 85 80 f5 ff 	vmovaps -0xa80(%rbp),%xmm0
  403feb:	ff 
  403fec:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  403ff0:	8b 45 30             	mov    0x30(%rbp),%eax
  403ff3:	c1 e0 02             	shl    $0x2,%eax
  403ff6:	0f af 45 10          	imul   0x10(%rbp),%eax
  403ffa:	8b 55 18             	mov    0x18(%rbp),%edx
  403ffd:	c1 e2 02             	shl    $0x2,%edx
  404000:	01 c2                	add    %eax,%edx
  404002:	8b 45 30             	mov    0x30(%rbp),%eax
  404005:	01 c0                	add    %eax,%eax
  404007:	01 d0                	add    %edx,%eax
  404009:	48 98                	cltq   
  40400b:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  404012:	00 
  404013:	48 8b 85 58 f2 ff ff 	mov    -0xda8(%rbp),%rax
  40401a:	48 01 d0             	add    %rdx,%rax
  40401d:	48 89 85 78 f5 ff ff 	mov    %rax,-0xa88(%rbp)
  404024:	c5 f8 28 85 c0 fe ff 	vmovaps -0x140(%rbp),%xmm0
  40402b:	ff 
  40402c:	c5 f8 29 85 60 f5 ff 	vmovaps %xmm0,-0xaa0(%rbp)
  404033:	ff 
  404034:	48 8b 85 78 f5 ff ff 	mov    -0xa88(%rbp),%rax
  40403b:	c5 f8 28 85 60 f5 ff 	vmovaps -0xaa0(%rbp),%xmm0
  404042:	ff 
  404043:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  404047:	8b 45 30             	mov    0x30(%rbp),%eax
  40404a:	c1 e0 02             	shl    $0x2,%eax
  40404d:	0f af 45 10          	imul   0x10(%rbp),%eax
  404051:	8b 55 18             	mov    0x18(%rbp),%edx
  404054:	c1 e2 02             	shl    $0x2,%edx
  404057:	01 c2                	add    %eax,%edx
  404059:	8b 45 30             	mov    0x30(%rbp),%eax
  40405c:	01 c0                	add    %eax,%eax
  40405e:	01 d0                	add    %edx,%eax
  404060:	48 98                	cltq   
  404062:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  404069:	00 
  40406a:	48 8b 85 50 f2 ff ff 	mov    -0xdb0(%rbp),%rax
  404071:	48 01 d0             	add    %rdx,%rax
  404074:	48 89 85 58 f5 ff ff 	mov    %rax,-0xaa8(%rbp)
  40407b:	c5 f8 28 85 b0 fe ff 	vmovaps -0x150(%rbp),%xmm0
  404082:	ff 
  404083:	c5 f8 29 85 40 f5 ff 	vmovaps %xmm0,-0xac0(%rbp)
  40408a:	ff 
  40408b:	48 8b 85 58 f5 ff ff 	mov    -0xaa8(%rbp),%rax
  404092:	c5 f8 28 85 40 f5 ff 	vmovaps -0xac0(%rbp),%xmm0
  404099:	ff 
  40409a:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  40409e:	48 8b 45 20          	mov    0x20(%rbp),%rax
  4040a2:	48 05 c0 00 00 00    	add    $0xc0,%rax
  4040a8:	48 89 85 38 f5 ff ff 	mov    %rax,-0xac8(%rbp)
  4040af:	48 8b 85 38 f5 ff ff 	mov    -0xac8(%rbp),%rax
  4040b6:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  4040ba:	c5 f8 29 85 70 fe ff 	vmovaps %xmm0,-0x190(%rbp)
  4040c1:	ff 
  4040c2:	48 8b 45 20          	mov    0x20(%rbp),%rax
  4040c6:	48 05 d0 00 00 00    	add    $0xd0,%rax
  4040cc:	48 89 85 30 f5 ff ff 	mov    %rax,-0xad0(%rbp)
  4040d3:	48 8b 85 30 f5 ff ff 	mov    -0xad0(%rbp),%rax
  4040da:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  4040de:	c5 f8 29 85 60 fe ff 	vmovaps %xmm0,-0x1a0(%rbp)
  4040e5:	ff 
  4040e6:	48 8b 45 20          	mov    0x20(%rbp),%rax
  4040ea:	48 05 e0 00 00 00    	add    $0xe0,%rax
  4040f0:	48 89 85 28 f5 ff ff 	mov    %rax,-0xad8(%rbp)
  4040f7:	48 8b 85 28 f5 ff ff 	mov    -0xad8(%rbp),%rax
  4040fe:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  404102:	c5 f8 29 85 50 fe ff 	vmovaps %xmm0,-0x1b0(%rbp)
  404109:	ff 
  40410a:	48 8b 45 20          	mov    0x20(%rbp),%rax
  40410e:	48 05 f0 00 00 00    	add    $0xf0,%rax
  404114:	48 89 85 20 f5 ff ff 	mov    %rax,-0xae0(%rbp)
  40411b:	48 8b 85 20 f5 ff ff 	mov    -0xae0(%rbp),%rax
  404122:	c5 f8 28 00          	vmovaps (%rax),%xmm0
  404126:	c5 f8 29 85 40 fe ff 	vmovaps %xmm0,-0x1c0(%rbp)
  40412d:	ff 
  40412e:	c5 f8 28 45 f0       	vmovaps -0x10(%rbp),%xmm0
  404133:	c5 f8 29 85 10 f5 ff 	vmovaps %xmm0,-0xaf0(%rbp)
  40413a:	ff 
  40413b:	c5 f8 28 85 70 fe ff 	vmovaps -0x190(%rbp),%xmm0
  404142:	ff 
  404143:	c5 f8 29 85 00 f5 ff 	vmovaps %xmm0,-0xb00(%rbp)
  40414a:	ff 
  40414b:	c5 f8 28 85 a0 fe ff 	vmovaps -0x160(%rbp),%xmm0
  404152:	ff 
  404153:	c5 f8 29 85 f0 f4 ff 	vmovaps %xmm0,-0xb10(%rbp)
  40415a:	ff 
  40415b:	c5 f8 28 8d 00 f5 ff 	vmovaps -0xb00(%rbp),%xmm1
  404162:	ff 
  404163:	c5 f8 28 85 f0 f4 ff 	vmovaps -0xb10(%rbp),%xmm0
  40416a:	ff 
  40416b:	c4 e2 71 b8 85 10 f5 	vfmadd231ps -0xaf0(%rbp),%xmm1,%xmm0
  404172:	ff ff 
  404174:	c5 f8 29 85 a0 fe ff 	vmovaps %xmm0,-0x160(%rbp)
  40417b:	ff 
  40417c:	c5 f8 28 45 e0       	vmovaps -0x20(%rbp),%xmm0
  404181:	c5 f8 29 85 e0 f4 ff 	vmovaps %xmm0,-0xb20(%rbp)
  404188:	ff 
  404189:	c5 f8 28 85 60 fe ff 	vmovaps -0x1a0(%rbp),%xmm0
  404190:	ff 
  404191:	c5 f8 29 85 d0 f4 ff 	vmovaps %xmm0,-0xb30(%rbp)
  404198:	ff 
  404199:	c5 f8 28 85 a0 fe ff 	vmovaps -0x160(%rbp),%xmm0
  4041a0:	ff 
  4041a1:	c5 f8 29 85 c0 f4 ff 	vmovaps %xmm0,-0xb40(%rbp)
  4041a8:	ff 
  4041a9:	c5 f8 28 8d d0 f4 ff 	vmovaps -0xb30(%rbp),%xmm1
  4041b0:	ff 
  4041b1:	c5 f8 28 85 c0 f4 ff 	vmovaps -0xb40(%rbp),%xmm0
  4041b8:	ff 
  4041b9:	c4 e2 71 b8 85 e0 f4 	vfmadd231ps -0xb20(%rbp),%xmm1,%xmm0
  4041c0:	ff ff 
  4041c2:	c5 f8 29 85 a0 fe ff 	vmovaps %xmm0,-0x160(%rbp)
  4041c9:	ff 
  4041ca:	c5 f8 28 45 d0       	vmovaps -0x30(%rbp),%xmm0
  4041cf:	c5 f8 29 85 b0 f4 ff 	vmovaps %xmm0,-0xb50(%rbp)
  4041d6:	ff 
  4041d7:	c5 f8 28 85 50 fe ff 	vmovaps -0x1b0(%rbp),%xmm0
  4041de:	ff 
  4041df:	c5 f8 29 85 a0 f4 ff 	vmovaps %xmm0,-0xb60(%rbp)
  4041e6:	ff 
  4041e7:	c5 f8 28 85 a0 fe ff 	vmovaps -0x160(%rbp),%xmm0
  4041ee:	ff 
  4041ef:	c5 f8 29 85 90 f4 ff 	vmovaps %xmm0,-0xb70(%rbp)
  4041f6:	ff 
  4041f7:	c5 f8 28 8d a0 f4 ff 	vmovaps -0xb60(%rbp),%xmm1
  4041fe:	ff 
  4041ff:	c5 f8 28 85 90 f4 ff 	vmovaps -0xb70(%rbp),%xmm0
  404206:	ff 
  404207:	c4 e2 71 b8 85 b0 f4 	vfmadd231ps -0xb50(%rbp),%xmm1,%xmm0
  40420e:	ff ff 
  404210:	c5 f8 29 85 a0 fe ff 	vmovaps %xmm0,-0x160(%rbp)
  404217:	ff 
  404218:	c5 f8 28 45 c0       	vmovaps -0x40(%rbp),%xmm0
  40421d:	c5 f8 29 85 80 f4 ff 	vmovaps %xmm0,-0xb80(%rbp)
  404224:	ff 
  404225:	c5 f8 28 85 40 fe ff 	vmovaps -0x1c0(%rbp),%xmm0
  40422c:	ff 
  40422d:	c5 f8 29 85 70 f4 ff 	vmovaps %xmm0,-0xb90(%rbp)
  404234:	ff 
  404235:	c5 f8 28 85 a0 fe ff 	vmovaps -0x160(%rbp),%xmm0
  40423c:	ff 
  40423d:	c5 f8 29 85 60 f4 ff 	vmovaps %xmm0,-0xba0(%rbp)
  404244:	ff 
  404245:	c5 f8 28 8d 70 f4 ff 	vmovaps -0xb90(%rbp),%xmm1
  40424c:	ff 
  40424d:	c5 f8 28 85 60 f4 ff 	vmovaps -0xba0(%rbp),%xmm0
  404254:	ff 
  404255:	c4 e2 71 b8 85 80 f4 	vfmadd231ps -0xb80(%rbp),%xmm1,%xmm0
  40425c:	ff ff 
  40425e:	c5 f8 29 85 a0 fe ff 	vmovaps %xmm0,-0x160(%rbp)
  404265:	ff 
  404266:	c5 f8 28 45 b0       	vmovaps -0x50(%rbp),%xmm0
  40426b:	c5 f8 29 85 50 f4 ff 	vmovaps %xmm0,-0xbb0(%rbp)
  404272:	ff 
  404273:	c5 f8 28 85 70 fe ff 	vmovaps -0x190(%rbp),%xmm0
  40427a:	ff 
  40427b:	c5 f8 29 85 40 f4 ff 	vmovaps %xmm0,-0xbc0(%rbp)
  404282:	ff 
  404283:	c5 f8 28 85 90 fe ff 	vmovaps -0x170(%rbp),%xmm0
  40428a:	ff 
  40428b:	c5 f8 29 85 30 f4 ff 	vmovaps %xmm0,-0xbd0(%rbp)
  404292:	ff 
  404293:	c5 f8 28 8d 40 f4 ff 	vmovaps -0xbc0(%rbp),%xmm1
  40429a:	ff 
  40429b:	c5 f8 28 85 30 f4 ff 	vmovaps -0xbd0(%rbp),%xmm0
  4042a2:	ff 
  4042a3:	c4 e2 71 b8 85 50 f4 	vfmadd231ps -0xbb0(%rbp),%xmm1,%xmm0
  4042aa:	ff ff 
  4042ac:	c5 f8 29 85 90 fe ff 	vmovaps %xmm0,-0x170(%rbp)
  4042b3:	ff 
  4042b4:	c5 f8 28 45 a0       	vmovaps -0x60(%rbp),%xmm0
  4042b9:	c5 f8 29 85 20 f4 ff 	vmovaps %xmm0,-0xbe0(%rbp)
  4042c0:	ff 
  4042c1:	c5 f8 28 85 60 fe ff 	vmovaps -0x1a0(%rbp),%xmm0
  4042c8:	ff 
  4042c9:	c5 f8 29 85 10 f4 ff 	vmovaps %xmm0,-0xbf0(%rbp)
  4042d0:	ff 
  4042d1:	c5 f8 28 85 90 fe ff 	vmovaps -0x170(%rbp),%xmm0
  4042d8:	ff 
  4042d9:	c5 f8 29 85 00 f4 ff 	vmovaps %xmm0,-0xc00(%rbp)
  4042e0:	ff 
  4042e1:	c5 f8 28 8d 10 f4 ff 	vmovaps -0xbf0(%rbp),%xmm1
  4042e8:	ff 
  4042e9:	c5 f8 28 85 00 f4 ff 	vmovaps -0xc00(%rbp),%xmm0
  4042f0:	ff 
  4042f1:	c4 e2 71 b8 85 20 f4 	vfmadd231ps -0xbe0(%rbp),%xmm1,%xmm0
  4042f8:	ff ff 
  4042fa:	c5 f8 29 85 90 fe ff 	vmovaps %xmm0,-0x170(%rbp)
  404301:	ff 
  404302:	c5 f8 28 45 90       	vmovaps -0x70(%rbp),%xmm0
  404307:	c5 f8 29 85 f0 f3 ff 	vmovaps %xmm0,-0xc10(%rbp)
  40430e:	ff 
  40430f:	c5 f8 28 85 50 fe ff 	vmovaps -0x1b0(%rbp),%xmm0
  404316:	ff 
  404317:	c5 f8 29 85 e0 f3 ff 	vmovaps %xmm0,-0xc20(%rbp)
  40431e:	ff 
  40431f:	c5 f8 28 85 90 fe ff 	vmovaps -0x170(%rbp),%xmm0
  404326:	ff 
  404327:	c5 f8 29 85 d0 f3 ff 	vmovaps %xmm0,-0xc30(%rbp)
  40432e:	ff 
  40432f:	c5 f8 28 8d e0 f3 ff 	vmovaps -0xc20(%rbp),%xmm1
  404336:	ff 
  404337:	c5 f8 28 85 d0 f3 ff 	vmovaps -0xc30(%rbp),%xmm0
  40433e:	ff 
  40433f:	c4 e2 71 b8 85 f0 f3 	vfmadd231ps -0xc10(%rbp),%xmm1,%xmm0
  404346:	ff ff 
  404348:	c5 f8 29 85 90 fe ff 	vmovaps %xmm0,-0x170(%rbp)
  40434f:	ff 
  404350:	c5 f8 28 45 80       	vmovaps -0x80(%rbp),%xmm0
  404355:	c5 f8 29 85 c0 f3 ff 	vmovaps %xmm0,-0xc40(%rbp)
  40435c:	ff 
  40435d:	c5 f8 28 85 40 fe ff 	vmovaps -0x1c0(%rbp),%xmm0
  404364:	ff 
  404365:	c5 f8 29 85 b0 f3 ff 	vmovaps %xmm0,-0xc50(%rbp)
  40436c:	ff 
  40436d:	c5 f8 28 85 90 fe ff 	vmovaps -0x170(%rbp),%xmm0
  404374:	ff 
  404375:	c5 f8 29 85 a0 f3 ff 	vmovaps %xmm0,-0xc60(%rbp)
  40437c:	ff 
  40437d:	c5 f8 28 8d b0 f3 ff 	vmovaps -0xc50(%rbp),%xmm1
  404384:	ff 
  404385:	c5 f8 28 85 a0 f3 ff 	vmovaps -0xc60(%rbp),%xmm0
  40438c:	ff 
  40438d:	c4 e2 71 b8 85 c0 f3 	vfmadd231ps -0xc40(%rbp),%xmm1,%xmm0
  404394:	ff ff 
  404396:	c5 f8 29 85 90 fe ff 	vmovaps %xmm0,-0x170(%rbp)
  40439d:	ff 
  40439e:	c5 f8 28 85 70 ff ff 	vmovaps -0x90(%rbp),%xmm0
  4043a5:	ff 
  4043a6:	c5 f8 29 85 90 f3 ff 	vmovaps %xmm0,-0xc70(%rbp)
  4043ad:	ff 
  4043ae:	c5 f8 28 85 70 fe ff 	vmovaps -0x190(%rbp),%xmm0
  4043b5:	ff 
  4043b6:	c5 f8 29 85 80 f3 ff 	vmovaps %xmm0,-0xc80(%rbp)
  4043bd:	ff 
  4043be:	c5 f8 28 85 80 fe ff 	vmovaps -0x180(%rbp),%xmm0
  4043c5:	ff 
  4043c6:	c5 f8 29 85 70 f3 ff 	vmovaps %xmm0,-0xc90(%rbp)
  4043cd:	ff 
  4043ce:	c5 f8 28 8d 80 f3 ff 	vmovaps -0xc80(%rbp),%xmm1
  4043d5:	ff 
  4043d6:	c5 f8 28 85 70 f3 ff 	vmovaps -0xc90(%rbp),%xmm0
  4043dd:	ff 
  4043de:	c4 e2 71 b8 85 90 f3 	vfmadd231ps -0xc70(%rbp),%xmm1,%xmm0
  4043e5:	ff ff 
  4043e7:	c5 f8 29 85 80 fe ff 	vmovaps %xmm0,-0x180(%rbp)
  4043ee:	ff 
  4043ef:	c5 f8 28 85 60 ff ff 	vmovaps -0xa0(%rbp),%xmm0
  4043f6:	ff 
  4043f7:	c5 f8 29 85 60 f3 ff 	vmovaps %xmm0,-0xca0(%rbp)
  4043fe:	ff 
  4043ff:	c5 f8 28 85 60 fe ff 	vmovaps -0x1a0(%rbp),%xmm0
  404406:	ff 
  404407:	c5 f8 29 85 50 f3 ff 	vmovaps %xmm0,-0xcb0(%rbp)
  40440e:	ff 
  40440f:	c5 f8 28 85 80 fe ff 	vmovaps -0x180(%rbp),%xmm0
  404416:	ff 
  404417:	c5 f8 29 85 40 f3 ff 	vmovaps %xmm0,-0xcc0(%rbp)
  40441e:	ff 
  40441f:	c5 f8 28 8d 50 f3 ff 	vmovaps -0xcb0(%rbp),%xmm1
  404426:	ff 
  404427:	c5 f8 28 85 40 f3 ff 	vmovaps -0xcc0(%rbp),%xmm0
  40442e:	ff 
  40442f:	c4 e2 71 b8 85 60 f3 	vfmadd231ps -0xca0(%rbp),%xmm1,%xmm0
  404436:	ff ff 
  404438:	c5 f8 29 85 80 fe ff 	vmovaps %xmm0,-0x180(%rbp)
  40443f:	ff 
  404440:	c5 f8 28 85 50 ff ff 	vmovaps -0xb0(%rbp),%xmm0
  404447:	ff 
  404448:	c5 f8 29 85 30 f3 ff 	vmovaps %xmm0,-0xcd0(%rbp)
  40444f:	ff 
  404450:	c5 f8 28 85 50 fe ff 	vmovaps -0x1b0(%rbp),%xmm0
  404457:	ff 
  404458:	c5 f8 29 85 20 f3 ff 	vmovaps %xmm0,-0xce0(%rbp)
  40445f:	ff 
  404460:	c5 f8 28 85 80 fe ff 	vmovaps -0x180(%rbp),%xmm0
  404467:	ff 
  404468:	c5 f8 29 85 10 f3 ff 	vmovaps %xmm0,-0xcf0(%rbp)
  40446f:	ff 
  404470:	c5 f8 28 8d 20 f3 ff 	vmovaps -0xce0(%rbp),%xmm1
  404477:	ff 
  404478:	c5 f8 28 85 10 f3 ff 	vmovaps -0xcf0(%rbp),%xmm0
  40447f:	ff 
  404480:	c4 e2 71 b8 85 30 f3 	vfmadd231ps -0xcd0(%rbp),%xmm1,%xmm0
  404487:	ff ff 
  404489:	c5 f8 29 85 80 fe ff 	vmovaps %xmm0,-0x180(%rbp)
  404490:	ff 
  404491:	c5 f8 28 85 40 ff ff 	vmovaps -0xc0(%rbp),%xmm0
  404498:	ff 
  404499:	c5 f8 29 85 00 f3 ff 	vmovaps %xmm0,-0xd00(%rbp)
  4044a0:	ff 
  4044a1:	c5 f8 28 85 40 fe ff 	vmovaps -0x1c0(%rbp),%xmm0
  4044a8:	ff 
  4044a9:	c5 f8 29 85 f0 f2 ff 	vmovaps %xmm0,-0xd10(%rbp)
  4044b0:	ff 
  4044b1:	c5 f8 28 85 80 fe ff 	vmovaps -0x180(%rbp),%xmm0
  4044b8:	ff 
  4044b9:	c5 f8 29 85 e0 f2 ff 	vmovaps %xmm0,-0xd20(%rbp)
  4044c0:	ff 
  4044c1:	c5 f8 28 8d f0 f2 ff 	vmovaps -0xd10(%rbp),%xmm1
  4044c8:	ff 
  4044c9:	c5 f8 28 85 e0 f2 ff 	vmovaps -0xd20(%rbp),%xmm0
  4044d0:	ff 
  4044d1:	c4 e2 71 b8 85 00 f3 	vfmadd231ps -0xd00(%rbp),%xmm1,%xmm0
  4044d8:	ff ff 
  4044da:	c5 f8 29 85 80 fe ff 	vmovaps %xmm0,-0x180(%rbp)
  4044e1:	ff 
  4044e2:	8b 45 30             	mov    0x30(%rbp),%eax
  4044e5:	c1 e0 02             	shl    $0x2,%eax
  4044e8:	0f af 45 10          	imul   0x10(%rbp),%eax
  4044ec:	8b 55 18             	mov    0x18(%rbp),%edx
  4044ef:	c1 e2 02             	shl    $0x2,%edx
  4044f2:	8d 0c 10             	lea    (%rax,%rdx,1),%ecx
  4044f5:	8b 55 30             	mov    0x30(%rbp),%edx
  4044f8:	89 d0                	mov    %edx,%eax
  4044fa:	01 c0                	add    %eax,%eax
  4044fc:	01 d0                	add    %edx,%eax
  4044fe:	01 c8                	add    %ecx,%eax
  404500:	48 98                	cltq   
  404502:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  404509:	00 
  40450a:	48 8b 85 60 f2 ff ff 	mov    -0xda0(%rbp),%rax
  404511:	48 01 d0             	add    %rdx,%rax
  404514:	48 89 85 d8 f2 ff ff 	mov    %rax,-0xd28(%rbp)
  40451b:	c5 f8 28 85 a0 fe ff 	vmovaps -0x160(%rbp),%xmm0
  404522:	ff 
  404523:	c5 f8 29 85 c0 f2 ff 	vmovaps %xmm0,-0xd40(%rbp)
  40452a:	ff 
  40452b:	48 8b 85 d8 f2 ff ff 	mov    -0xd28(%rbp),%rax
  404532:	c5 f8 28 85 c0 f2 ff 	vmovaps -0xd40(%rbp),%xmm0
  404539:	ff 
  40453a:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  40453e:	8b 45 30             	mov    0x30(%rbp),%eax
  404541:	c1 e0 02             	shl    $0x2,%eax
  404544:	0f af 45 10          	imul   0x10(%rbp),%eax
  404548:	8b 55 18             	mov    0x18(%rbp),%edx
  40454b:	c1 e2 02             	shl    $0x2,%edx
  40454e:	8d 0c 10             	lea    (%rax,%rdx,1),%ecx
  404551:	8b 55 30             	mov    0x30(%rbp),%edx
  404554:	89 d0                	mov    %edx,%eax
  404556:	01 c0                	add    %eax,%eax
  404558:	01 d0                	add    %edx,%eax
  40455a:	01 c8                	add    %ecx,%eax
  40455c:	48 98                	cltq   
  40455e:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  404565:	00 
  404566:	48 8b 85 58 f2 ff ff 	mov    -0xda8(%rbp),%rax
  40456d:	48 01 d0             	add    %rdx,%rax
  404570:	48 89 85 b8 f2 ff ff 	mov    %rax,-0xd48(%rbp)
  404577:	c5 f8 28 85 90 fe ff 	vmovaps -0x170(%rbp),%xmm0
  40457e:	ff 
  40457f:	c5 f8 29 85 a0 f2 ff 	vmovaps %xmm0,-0xd60(%rbp)
  404586:	ff 
  404587:	48 8b 85 b8 f2 ff ff 	mov    -0xd48(%rbp),%rax
  40458e:	c5 f8 28 85 a0 f2 ff 	vmovaps -0xd60(%rbp),%xmm0
  404595:	ff 
  404596:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  40459a:	8b 45 30             	mov    0x30(%rbp),%eax
  40459d:	c1 e0 02             	shl    $0x2,%eax
  4045a0:	0f af 45 10          	imul   0x10(%rbp),%eax
  4045a4:	8b 55 18             	mov    0x18(%rbp),%edx
  4045a7:	c1 e2 02             	shl    $0x2,%edx
  4045aa:	8d 0c 10             	lea    (%rax,%rdx,1),%ecx
  4045ad:	8b 55 30             	mov    0x30(%rbp),%edx
  4045b0:	89 d0                	mov    %edx,%eax
  4045b2:	01 c0                	add    %eax,%eax
  4045b4:	01 d0                	add    %edx,%eax
  4045b6:	01 c8                	add    %ecx,%eax
  4045b8:	48 98                	cltq   
  4045ba:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  4045c1:	00 
  4045c2:	48 8b 85 50 f2 ff ff 	mov    -0xdb0(%rbp),%rax
  4045c9:	48 01 d0             	add    %rdx,%rax
  4045cc:	48 89 85 98 f2 ff ff 	mov    %rax,-0xd68(%rbp)
  4045d3:	c5 f8 28 85 80 fe ff 	vmovaps -0x180(%rbp),%xmm0
  4045da:	ff 
  4045db:	c5 f8 29 85 80 f2 ff 	vmovaps %xmm0,-0xd80(%rbp)
  4045e2:	ff 
  4045e3:	48 8b 85 98 f2 ff ff 	mov    -0xd68(%rbp),%rax
  4045ea:	c5 f8 28 85 80 f2 ff 	vmovaps -0xd80(%rbp),%xmm0
  4045f1:	ff 
  4045f2:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
  4045f6:	c9                   	leaveq 
  4045f7:	c3                   	retq   

00000000004045f8 <main>:
  4045f8:	55                   	push   %rbp
  4045f9:	48 89 e5             	mov    %rsp,%rbp
  4045fc:	41 54                	push   %r12
  4045fe:	53                   	push   %rbx
  4045ff:	48 81 ec 90 03 00 00 	sub    $0x390,%rsp
  404606:	89 bd 9c fc ff ff    	mov    %edi,-0x364(%rbp)
  40460c:	48 89 b5 90 fc ff ff 	mov    %rsi,-0x370(%rbp)
  404613:	48 8d 85 a0 fc ff ff 	lea    -0x360(%rbp),%rax
  40461a:	bb 00 00 00 00       	mov    $0x0,%ebx
  40461f:	49 89 c4             	mov    %rax,%r12
  404622:	eb 10                	jmp    404634 <main+0x3c>
  404624:	4c 89 e7             	mov    %r12,%rdi
  404627:	e8 d6 08 00 00       	callq  404f02 <_ZN2cv3MatC1Ev>
  40462c:	49 83 c4 60          	add    $0x60,%r12
  404630:	48 83 eb 01          	sub    $0x1,%rbx
  404634:	48 83 fb ff          	cmp    $0xffffffffffffffff,%rbx
  404638:	75 ea                	jne    404624 <main+0x2c>
  40463a:	48 8d 85 00 fd ff ff 	lea    -0x300(%rbp),%rax
  404641:	48 89 c7             	mov    %rax,%rdi
  404644:	e8 b9 08 00 00       	callq  404f02 <_ZN2cv3MatC1Ev>
  404649:	c7 45 ec 00 00 00 00 	movl   $0x0,-0x14(%rbp)
  404650:	e9 b4 00 00 00       	jmpq   404709 <main+0x111>
  404655:	48 8d 85 3f ff ff ff 	lea    -0xc1(%rbp),%rax
  40465c:	48 89 c7             	mov    %rax,%rdi
  40465f:	e8 ac d2 ff ff       	callq  401910 <_ZNSaIcEC1Ev@plt>
  404664:	8b 45 ec             	mov    -0x14(%rbp),%eax
  404667:	48 98                	cltq   
  404669:	48 8b 0c c5 a0 91 60 	mov    0x6091a0(,%rax,8),%rcx
  404670:	00 
  404671:	48 8d 95 3f ff ff ff 	lea    -0xc1(%rbp),%rdx
  404678:	48 8d 85 30 ff ff ff 	lea    -0xd0(%rbp),%rax
  40467f:	48 89 ce             	mov    %rcx,%rsi
  404682:	48 89 c7             	mov    %rax,%rdi
  404685:	e8 06 d3 ff ff       	callq  401990 <_ZNSsC1EPKcRKSaIcE@plt>
  40468a:	48 8d 85 d0 fe ff ff 	lea    -0x130(%rbp),%rax
  404691:	48 8d 8d 30 ff ff ff 	lea    -0xd0(%rbp),%rcx
  404698:	ba 01 00 00 00       	mov    $0x1,%edx
  40469d:	48 89 ce             	mov    %rcx,%rsi
  4046a0:	48 89 c7             	mov    %rax,%rdi
  4046a3:	e8 b8 d2 ff ff       	callq  401960 <_ZN2cv6imreadERKSsi@plt>
  4046a8:	48 8d 8d a0 fc ff ff 	lea    -0x360(%rbp),%rcx
  4046af:	8b 45 ec             	mov    -0x14(%rbp),%eax
  4046b2:	48 63 d0             	movslq %eax,%rdx
  4046b5:	48 89 d0             	mov    %rdx,%rax
  4046b8:	48 01 c0             	add    %rax,%rax
  4046bb:	48 01 d0             	add    %rdx,%rax
  4046be:	48 c1 e0 05          	shl    $0x5,%rax
  4046c2:	48 8d 14 01          	lea    (%rcx,%rax,1),%rdx
  4046c6:	48 8d 85 d0 fe ff ff 	lea    -0x130(%rbp),%rax
  4046cd:	48 89 c6             	mov    %rax,%rsi
  4046d0:	48 89 d7             	mov    %rdx,%rdi
  4046d3:	e8 24 0c 00 00       	callq  4052fc <_ZN2cv3MataSERKS0_>
  4046d8:	48 8d 85 d0 fe ff ff 	lea    -0x130(%rbp),%rax
  4046df:	48 89 c7             	mov    %rax,%rdi
  4046e2:	e8 d5 0b 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  4046e7:	48 8d 85 30 ff ff ff 	lea    -0xd0(%rbp),%rax
  4046ee:	48 89 c7             	mov    %rax,%rdi
  4046f1:	e8 8a d2 ff ff       	callq  401980 <_ZNSsD1Ev@plt>
  4046f6:	48 8d 85 3f ff ff ff 	lea    -0xc1(%rbp),%rax
  4046fd:	48 89 c7             	mov    %rax,%rdi
  404700:	e8 ab d2 ff ff       	callq  4019b0 <_ZNSaIcED1Ev@plt>
  404705:	83 45 ec 01          	addl   $0x1,-0x14(%rbp)
  404709:	83 7d ec 01          	cmpl   $0x1,-0x14(%rbp)
  40470d:	0f 8e 42 ff ff ff    	jle    404655 <main+0x5d>
  404713:	be 04 00 00 00       	mov    $0x4,%esi
  404718:	bf 40 00 00 00       	mov    $0x40,%edi
  40471d:	e8 9e d1 ff ff       	callq  4018c0 <calloc@plt>
  404722:	48 89 45 d0          	mov    %rax,-0x30(%rbp)
  404726:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  40472a:	48 89 c7             	mov    %rax,%rdi
  40472d:	e8 4b db ff ff       	callq  40227d <_Z20generateCoefficientsPf>
  404732:	c7 45 e8 00 00 00 00 	movl   $0x0,-0x18(%rbp)
  404739:	e9 15 05 00 00       	jmpq   404c53 <main+0x65b>
  40473e:	48 8d 8d a0 fc ff ff 	lea    -0x360(%rbp),%rcx
  404745:	8b 45 e8             	mov    -0x18(%rbp),%eax
  404748:	48 63 d0             	movslq %eax,%rdx
  40474b:	48 89 d0             	mov    %rdx,%rax
  40474e:	48 01 c0             	add    %rax,%rax
  404751:	48 01 d0             	add    %rdx,%rax
  404754:	48 c1 e0 05          	shl    $0x5,%rax
  404758:	48 8d 14 01          	lea    (%rcx,%rax,1),%rdx
  40475c:	48 8d 85 00 fd ff ff 	lea    -0x300(%rbp),%rax
  404763:	48 89 d6             	mov    %rdx,%rsi
  404766:	48 89 c7             	mov    %rax,%rdi
  404769:	e8 8e 0b 00 00       	callq  4052fc <_ZN2cv3MataSERKS0_>
  40476e:	8b 85 0c fd ff ff    	mov    -0x2f4(%rbp),%eax
  404774:	89 45 cc             	mov    %eax,-0x34(%rbp)
  404777:	8b 85 08 fd ff ff    	mov    -0x2f8(%rbp),%eax
  40477d:	89 45 c8             	mov    %eax,-0x38(%rbp)
  404780:	8b 45 c8             	mov    -0x38(%rbp),%eax
  404783:	0f af 45 cc          	imul   -0x34(%rbp),%eax
  404787:	48 98                	cltq   
  404789:	be 04 00 00 00       	mov    $0x4,%esi
  40478e:	48 89 c7             	mov    %rax,%rdi
  404791:	e8 2a d1 ff ff       	callq  4018c0 <calloc@plt>
  404796:	48 89 45 c0          	mov    %rax,-0x40(%rbp)
  40479a:	8b 45 c8             	mov    -0x38(%rbp),%eax
  40479d:	0f af 45 cc          	imul   -0x34(%rbp),%eax
  4047a1:	48 98                	cltq   
  4047a3:	be 04 00 00 00       	mov    $0x4,%esi
  4047a8:	48 89 c7             	mov    %rax,%rdi
  4047ab:	e8 10 d1 ff ff       	callq  4018c0 <calloc@plt>
  4047b0:	48 89 45 b8          	mov    %rax,-0x48(%rbp)
  4047b4:	8b 45 c8             	mov    -0x38(%rbp),%eax
  4047b7:	0f af 45 cc          	imul   -0x34(%rbp),%eax
  4047bb:	48 98                	cltq   
  4047bd:	be 04 00 00 00       	mov    $0x4,%esi
  4047c2:	48 89 c7             	mov    %rax,%rdi
  4047c5:	e8 f6 d0 ff ff       	callq  4018c0 <calloc@plt>
  4047ca:	48 89 45 b0          	mov    %rax,-0x50(%rbp)
  4047ce:	8b 45 e8             	mov    -0x18(%rbp),%eax
  4047d1:	48 98                	cltq   
  4047d3:	48 8b 1c c5 a0 91 60 	mov    0x6091a0(,%rax,8),%rbx
  4047da:	00 
  4047db:	be 02 71 40 00       	mov    $0x407102,%esi
  4047e0:	bf c0 91 60 00       	mov    $0x6091c0,%edi
  4047e5:	e8 f6 cf ff ff       	callq  4017e0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
  4047ea:	48 89 de             	mov    %rbx,%rsi
  4047ed:	48 89 c7             	mov    %rax,%rdi
  4047f0:	e8 eb cf ff ff       	callq  4017e0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
  4047f5:	be 20 19 40 00       	mov    $0x401920,%esi
  4047fa:	48 89 c7             	mov    %rax,%rdi
  4047fd:	e8 4e d1 ff ff       	callq  401950 <_ZNSolsEPFRSoS_E@plt>
  404802:	8b 45 e8             	mov    -0x18(%rbp),%eax
  404805:	48 98                	cltq   
  404807:	48 8b 0c c5 a0 91 60 	mov    0x6091a0(,%rax,8),%rcx
  40480e:	00 
  40480f:	48 8b 55 b8          	mov    -0x48(%rbp),%rdx
  404813:	48 8b 75 b0          	mov    -0x50(%rbp),%rsi
  404817:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
  40481b:	48 89 c7             	mov    %rax,%rdi
  40481e:	e8 a5 d2 ff ff       	callq  401ac8 <_Z11decodeImagePfS_S_PKc>
  404823:	c7 45 e4 00 00 00 00 	movl   $0x0,-0x1c(%rbp)
  40482a:	e9 f2 03 00 00       	jmpq   404c21 <main+0x629>
  40482f:	8b 45 e4             	mov    -0x1c(%rbp),%eax
  404832:	48 98                	cltq   
  404834:	8b 04 85 b0 91 60 00 	mov    0x6091b0(,%rax,4),%eax
  40483b:	0f af 45 cc          	imul   -0x34(%rbp),%eax
  40483f:	89 45 ac             	mov    %eax,-0x54(%rbp)
  404842:	8b 45 e4             	mov    -0x1c(%rbp),%eax
  404845:	48 98                	cltq   
  404847:	8b 04 85 b0 91 60 00 	mov    0x6091b0(,%rax,4),%eax
  40484e:	0f af 45 c8          	imul   -0x38(%rbp),%eax
  404852:	89 45 a8             	mov    %eax,-0x58(%rbp)
  404855:	8b 45 a8             	mov    -0x58(%rbp),%eax
  404858:	0f af 45 ac          	imul   -0x54(%rbp),%eax
  40485c:	48 98                	cltq   
  40485e:	be 04 00 00 00       	mov    $0x4,%esi
  404863:	48 89 c7             	mov    %rax,%rdi
  404866:	e8 55 d0 ff ff       	callq  4018c0 <calloc@plt>
  40486b:	48 89 45 a0          	mov    %rax,-0x60(%rbp)
  40486f:	8b 45 a8             	mov    -0x58(%rbp),%eax
  404872:	0f af 45 ac          	imul   -0x54(%rbp),%eax
  404876:	48 98                	cltq   
  404878:	be 04 00 00 00       	mov    $0x4,%esi
  40487d:	48 89 c7             	mov    %rax,%rdi
  404880:	e8 3b d0 ff ff       	callq  4018c0 <calloc@plt>
  404885:	48 89 45 98          	mov    %rax,-0x68(%rbp)
  404889:	8b 45 a8             	mov    -0x58(%rbp),%eax
  40488c:	0f af 45 ac          	imul   -0x54(%rbp),%eax
  404890:	48 98                	cltq   
  404892:	be 04 00 00 00       	mov    $0x4,%esi
  404897:	48 89 c7             	mov    %rax,%rdi
  40489a:	e8 21 d0 ff ff       	callq  4018c0 <calloc@plt>
  40489f:	48 89 45 90          	mov    %rax,-0x70(%rbp)
  4048a3:	e8 05 d2 ff ff       	callq  401aad <_ZL5rdtscv>
  4048a8:	48 89 45 88          	mov    %rax,-0x78(%rbp)
  4048ac:	c7 45 e0 00 00 00 00 	movl   $0x0,-0x20(%rbp)
  4048b3:	eb 68                	jmp    40491d <main+0x325>
  4048b5:	c7 45 dc 00 00 00 00 	movl   $0x0,-0x24(%rbp)
  4048bc:	eb 48                	jmp    404906 <main+0x30e>
  4048be:	4c 8b 4d 90          	mov    -0x70(%rbp),%r9
  4048c2:	4c 8b 45 98          	mov    -0x68(%rbp),%r8
  4048c6:	48 8b 4d a0          	mov    -0x60(%rbp),%rcx
  4048ca:	48 8b 55 b0          	mov    -0x50(%rbp),%rdx
  4048ce:	48 8b 75 b8          	mov    -0x48(%rbp),%rsi
  4048d2:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
  4048d6:	8b 7d ac             	mov    -0x54(%rbp),%edi
  4048d9:	89 7c 24 20          	mov    %edi,0x20(%rsp)
  4048dd:	8b 7d a8             	mov    -0x58(%rbp),%edi
  4048e0:	89 7c 24 18          	mov    %edi,0x18(%rsp)
  4048e4:	48 8b 7d d0          	mov    -0x30(%rbp),%rdi
  4048e8:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4048ed:	8b 7d dc             	mov    -0x24(%rbp),%edi
  4048f0:	89 7c 24 08          	mov    %edi,0x8(%rsp)
  4048f4:	8b 7d e0             	mov    -0x20(%rbp),%edi
  4048f7:	89 3c 24             	mov    %edi,(%rsp)
  4048fa:	48 89 c7             	mov    %rax,%rdi
  4048fd:	e8 75 e1 ff ff       	callq  402a77 <_Z6kernelPKfS0_S0_PfS1_S1_iiS0_ii>
  404902:	83 45 dc 01          	addl   $0x1,-0x24(%rbp)
  404906:	8b 45 ac             	mov    -0x54(%rbp),%eax
  404909:	8d 50 03             	lea    0x3(%rax),%edx
  40490c:	85 c0                	test   %eax,%eax
  40490e:	0f 48 c2             	cmovs  %edx,%eax
  404911:	c1 f8 02             	sar    $0x2,%eax
  404914:	3b 45 dc             	cmp    -0x24(%rbp),%eax
  404917:	7f a5                	jg     4048be <main+0x2c6>
  404919:	83 45 e0 01          	addl   $0x1,-0x20(%rbp)
  40491d:	8b 45 a8             	mov    -0x58(%rbp),%eax
  404920:	8d 50 03             	lea    0x3(%rax),%edx
  404923:	85 c0                	test   %eax,%eax
  404925:	0f 48 c2             	cmovs  %edx,%eax
  404928:	c1 f8 02             	sar    $0x2,%eax
  40492b:	3b 45 e0             	cmp    -0x20(%rbp),%eax
  40492e:	7f 85                	jg     4048b5 <main+0x2bd>
  404930:	e8 78 d1 ff ff       	callq  401aad <_ZL5rdtscv>
  404935:	48 89 45 80          	mov    %rax,-0x80(%rbp)
  404939:	48 8d 85 60 fd ff ff 	lea    -0x2a0(%rbp),%rax
  404940:	be 10 00 00 00       	mov    $0x10,%esi
  404945:	48 89 c7             	mov    %rax,%rdi
  404948:	e8 a3 ce ff ff       	callq  4017f0 <_ZNSt19basic_ostringstreamIcSt11char_traitsIcESaIcEEC1ESt13_Ios_Openmode@plt>
  40494d:	48 8d 85 4f ff ff ff 	lea    -0xb1(%rbp),%rax
  404954:	48 89 c7             	mov    %rax,%rdi
  404957:	e8 b4 cf ff ff       	callq  401910 <_ZNSaIcEC1Ev@plt>
  40495c:	48 8d 95 4f ff ff ff 	lea    -0xb1(%rbp),%rdx
  404963:	48 8d 85 40 ff ff ff 	lea    -0xc0(%rbp),%rax
  40496a:	be 0a 71 40 00       	mov    $0x40710a,%esi
  40496f:	48 89 c7             	mov    %rax,%rdi
  404972:	e8 19 d0 ff ff       	callq  401990 <_ZNSsC1EPKcRKSaIcE@plt>
  404977:	48 8d 95 40 ff ff ff 	lea    -0xc0(%rbp),%rdx
  40497e:	48 8d 85 60 fd ff ff 	lea    -0x2a0(%rbp),%rax
  404985:	48 89 d6             	mov    %rdx,%rsi
  404988:	48 89 c7             	mov    %rax,%rdi
  40498b:	e8 10 ce ff ff       	callq  4017a0 <_ZNSt19basic_ostringstreamIcSt11char_traitsIcESaIcEE3strERKSs@plt>
  404990:	48 8d 85 40 ff ff ff 	lea    -0xc0(%rbp),%rax
  404997:	48 89 c7             	mov    %rax,%rdi
  40499a:	e8 e1 cf ff ff       	callq  401980 <_ZNSsD1Ev@plt>
  40499f:	48 8d 85 4f ff ff ff 	lea    -0xb1(%rbp),%rax
  4049a6:	48 89 c7             	mov    %rax,%rdi
  4049a9:	e8 02 d0 ff ff       	callq  4019b0 <_ZNSaIcED1Ev@plt>
  4049ae:	48 8d 85 50 ff ff ff 	lea    -0xb0(%rbp),%rax
  4049b5:	48 8d 95 60 fd ff ff 	lea    -0x2a0(%rbp),%rdx
  4049bc:	48 89 d6             	mov    %rdx,%rsi
  4049bf:	48 89 c7             	mov    %rax,%rdi
  4049c2:	e8 59 ce ff ff       	callq  401820 <_ZNKSt19basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv@plt>
  4049c7:	48 8d 85 50 ff ff ff 	lea    -0xb0(%rbp),%rax
  4049ce:	48 89 c7             	mov    %rax,%rdi
  4049d1:	e8 ca ce ff ff       	callq  4018a0 <_ZNSs5clearEv@plt>
  4049d6:	48 8d 85 50 ff ff ff 	lea    -0xb0(%rbp),%rax
  4049dd:	48 89 c7             	mov    %rax,%rdi
  4049e0:	e8 9b cf ff ff       	callq  401980 <_ZNSsD1Ev@plt>
  4049e5:	48 8b 15 a4 47 20 00 	mov    0x2047a4(%rip),%rdx        # 609190 <home>
  4049ec:	48 8d 85 60 fd ff ff 	lea    -0x2a0(%rbp),%rax
  4049f3:	48 89 d6             	mov    %rdx,%rsi
  4049f6:	48 89 c7             	mov    %rax,%rdi
  4049f9:	e8 e2 cd ff ff       	callq  4017e0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
  4049fe:	be 0b 71 40 00       	mov    $0x40710b,%esi
  404a03:	48 89 c7             	mov    %rax,%rdi
  404a06:	e8 d5 cd ff ff       	callq  4017e0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
  404a0b:	8b 55 cc             	mov    -0x34(%rbp),%edx
  404a0e:	89 d6                	mov    %edx,%esi
  404a10:	48 89 c7             	mov    %rax,%rdi
  404a13:	e8 78 cd ff ff       	callq  401790 <_ZNSolsEi@plt>
  404a18:	be 1a 71 40 00       	mov    $0x40711a,%esi
  404a1d:	48 89 c7             	mov    %rax,%rdi
  404a20:	e8 bb cd ff ff       	callq  4017e0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
  404a25:	8b 55 c8             	mov    -0x38(%rbp),%edx
  404a28:	89 d6                	mov    %edx,%esi
  404a2a:	48 89 c7             	mov    %rax,%rdi
  404a2d:	e8 5e cd ff ff       	callq  401790 <_ZNSolsEi@plt>
  404a32:	be 1c 71 40 00       	mov    $0x40711c,%esi
  404a37:	48 89 c7             	mov    %rax,%rdi
  404a3a:	e8 a1 cd ff ff       	callq  4017e0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
  404a3f:	8b 55 ac             	mov    -0x54(%rbp),%edx
  404a42:	89 d6                	mov    %edx,%esi
  404a44:	48 89 c7             	mov    %rax,%rdi
  404a47:	e8 44 cd ff ff       	callq  401790 <_ZNSolsEi@plt>
  404a4c:	be 1a 71 40 00       	mov    $0x40711a,%esi
  404a51:	48 89 c7             	mov    %rax,%rdi
  404a54:	e8 87 cd ff ff       	callq  4017e0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
  404a59:	8b 55 a8             	mov    -0x58(%rbp),%edx
  404a5c:	89 d6                	mov    %edx,%esi
  404a5e:	48 89 c7             	mov    %rax,%rdi
  404a61:	e8 2a cd ff ff       	callq  401790 <_ZNSolsEi@plt>
  404a66:	be 1e 71 40 00       	mov    $0x40711e,%esi
  404a6b:	48 89 c7             	mov    %rax,%rdi
  404a6e:	e8 6d cd ff ff       	callq  4017e0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
  404a73:	48 8d 85 c0 fe ff ff 	lea    -0x140(%rbp),%rax
  404a7a:	48 8d 95 60 fd ff ff 	lea    -0x2a0(%rbp),%rdx
  404a81:	48 89 d6             	mov    %rdx,%rsi
  404a84:	48 89 c7             	mov    %rax,%rdi
  404a87:	e8 94 cd ff ff       	callq  401820 <_ZNKSt19basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv@plt>
  404a8c:	be 23 71 40 00       	mov    $0x407123,%esi
  404a91:	bf c0 91 60 00       	mov    $0x6091c0,%edi
  404a96:	e8 45 cd ff ff       	callq  4017e0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
  404a9b:	48 8d 95 c0 fe ff ff 	lea    -0x140(%rbp),%rdx
  404aa2:	48 89 d6             	mov    %rdx,%rsi
  404aa5:	48 89 c7             	mov    %rax,%rdi
  404aa8:	e8 f3 ce ff ff       	callq  4019a0 <_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E@plt>
  404aad:	be 20 19 40 00       	mov    $0x401920,%esi
  404ab2:	48 89 c7             	mov    %rax,%rdi
  404ab5:	e8 96 ce ff ff       	callq  401950 <_ZNSolsEPFRSoS_E@plt>
  404aba:	48 8d 85 c0 fe ff ff 	lea    -0x140(%rbp),%rax
  404ac1:	48 89 c7             	mov    %rax,%rdi
  404ac4:	e8 a7 ce ff ff       	callq  401970 <_ZNKSs5c_strEv@plt>
  404ac9:	48 89 c1             	mov    %rax,%rcx
  404acc:	44 8b 45 a8          	mov    -0x58(%rbp),%r8d
  404ad0:	8b 7d ac             	mov    -0x54(%rbp),%edi
  404ad3:	48 8b 55 90          	mov    -0x70(%rbp),%rdx
  404ad7:	48 8b 75 98          	mov    -0x68(%rbp),%rsi
  404adb:	48 8b 45 a0          	mov    -0x60(%rbp),%rax
  404adf:	45 89 c1             	mov    %r8d,%r9d
  404ae2:	41 89 f8             	mov    %edi,%r8d
  404ae5:	48 89 c7             	mov    %rax,%rdi
  404ae8:	e8 86 d4 ff ff       	callq  401f73 <_Z11encodeImagePfS_S_PKcii>
  404aed:	48 8d 85 6f ff ff ff 	lea    -0x91(%rbp),%rax
  404af4:	48 89 c7             	mov    %rax,%rdi
  404af7:	e8 14 ce ff ff       	callq  401910 <_ZNSaIcEC1Ev@plt>
  404afc:	48 8d 95 6f ff ff ff 	lea    -0x91(%rbp),%rdx
  404b03:	48 8d 85 60 ff ff ff 	lea    -0xa0(%rbp),%rax
  404b0a:	be 0a 71 40 00       	mov    $0x40710a,%esi
  404b0f:	48 89 c7             	mov    %rax,%rdi
  404b12:	e8 79 ce ff ff       	callq  401990 <_ZNSsC1EPKcRKSaIcE@plt>
  404b17:	48 8d 95 60 ff ff ff 	lea    -0xa0(%rbp),%rdx
  404b1e:	48 8d 85 60 fd ff ff 	lea    -0x2a0(%rbp),%rax
  404b25:	48 89 d6             	mov    %rdx,%rsi
  404b28:	48 89 c7             	mov    %rax,%rdi
  404b2b:	e8 70 cc ff ff       	callq  4017a0 <_ZNSt19basic_ostringstreamIcSt11char_traitsIcESaIcEE3strERKSs@plt>
  404b30:	48 8d 85 60 ff ff ff 	lea    -0xa0(%rbp),%rax
  404b37:	48 89 c7             	mov    %rax,%rdi
  404b3a:	e8 41 ce ff ff       	callq  401980 <_ZNSsD1Ev@plt>
  404b3f:	48 8d 85 6f ff ff ff 	lea    -0x91(%rbp),%rax
  404b46:	48 89 c7             	mov    %rax,%rdi
  404b49:	e8 62 ce ff ff       	callq  4019b0 <_ZNSaIcED1Ev@plt>
  404b4e:	48 8d 85 70 ff ff ff 	lea    -0x90(%rbp),%rax
  404b55:	48 8d 95 60 fd ff ff 	lea    -0x2a0(%rbp),%rdx
  404b5c:	48 89 d6             	mov    %rdx,%rsi
  404b5f:	48 89 c7             	mov    %rax,%rdi
  404b62:	e8 b9 cc ff ff       	callq  401820 <_ZNKSt19basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv@plt>
  404b67:	48 8d 85 70 ff ff ff 	lea    -0x90(%rbp),%rax
  404b6e:	48 89 c7             	mov    %rax,%rdi
  404b71:	e8 2a cd ff ff       	callq  4018a0 <_ZNSs5clearEv@plt>
  404b76:	48 8d 85 70 ff ff ff 	lea    -0x90(%rbp),%rax
  404b7d:	48 89 c7             	mov    %rax,%rdi
  404b80:	e8 fb cd ff ff       	callq  401980 <_ZNSsD1Ev@plt>
  404b85:	48 8b 45 88          	mov    -0x78(%rbp),%rax
  404b89:	48 8b 55 80          	mov    -0x80(%rbp),%rdx
  404b8d:	48 29 c2             	sub    %rax,%rdx
  404b90:	48 89 d0             	mov    %rdx,%rax
  404b93:	48 85 c0             	test   %rax,%rax
  404b96:	78 07                	js     404b9f <main+0x5a7>
  404b98:	c4 e1 fb 2a c0       	vcvtsi2sd %rax,%xmm0,%xmm0
  404b9d:	eb 15                	jmp    404bb4 <main+0x5bc>
  404b9f:	48 89 c2             	mov    %rax,%rdx
  404ba2:	48 d1 ea             	shr    %rdx
  404ba5:	83 e0 01             	and    $0x1,%eax
  404ba8:	48 09 c2             	or     %rax,%rdx
  404bab:	c4 e1 fb 2a c2       	vcvtsi2sd %rdx,%xmm0,%xmm0
  404bb0:	c5 fb 58 c0          	vaddsd %xmm0,%xmm0,%xmm0
  404bb4:	c5 fb 10 0d d4 25 00 	vmovsd 0x25d4(%rip),%xmm1        # 407190 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x30>
  404bbb:	00 
  404bbc:	c5 fb 59 c1          	vmulsd %xmm1,%xmm0,%xmm0
  404bc0:	c5 fb 10 0d d0 25 00 	vmovsd 0x25d0(%rip),%xmm1        # 407198 <_ZZL18__gthread_active_pvE20__gthread_active_ptr+0x38>
  404bc7:	00 
  404bc8:	c5 fb 5e c1          	vdivsd %xmm1,%xmm0,%xmm0
  404bcc:	bf 2c 71 40 00       	mov    $0x40712c,%edi
  404bd1:	b8 01 00 00 00       	mov    $0x1,%eax
  404bd6:	e8 35 cb ff ff       	callq  401710 <printf@plt>
  404bdb:	48 8b 45 a0          	mov    -0x60(%rbp),%rax
  404bdf:	48 89 c7             	mov    %rax,%rdi
  404be2:	e8 89 cb ff ff       	callq  401770 <free@plt>
  404be7:	48 8b 45 98          	mov    -0x68(%rbp),%rax
  404beb:	48 89 c7             	mov    %rax,%rdi
  404bee:	e8 7d cb ff ff       	callq  401770 <free@plt>
  404bf3:	48 8b 45 90          	mov    -0x70(%rbp),%rax
  404bf7:	48 89 c7             	mov    %rax,%rdi
  404bfa:	e8 71 cb ff ff       	callq  401770 <free@plt>
  404bff:	48 8d 85 c0 fe ff ff 	lea    -0x140(%rbp),%rax
  404c06:	48 89 c7             	mov    %rax,%rdi
  404c09:	e8 72 cd ff ff       	callq  401980 <_ZNSsD1Ev@plt>
  404c0e:	48 8d 85 60 fd ff ff 	lea    -0x2a0(%rbp),%rax
  404c15:	48 89 c7             	mov    %rax,%rdi
  404c18:	e8 33 cb ff ff       	callq  401750 <_ZNSt19basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev@plt>
  404c1d:	83 45 e4 01          	addl   $0x1,-0x1c(%rbp)
  404c21:	83 7d e4 00          	cmpl   $0x0,-0x1c(%rbp)
  404c25:	0f 8e 04 fc ff ff    	jle    40482f <main+0x237>
  404c2b:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
  404c2f:	48 89 c7             	mov    %rax,%rdi
  404c32:	e8 39 cb ff ff       	callq  401770 <free@plt>
  404c37:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
  404c3b:	48 89 c7             	mov    %rax,%rdi
  404c3e:	e8 2d cb ff ff       	callq  401770 <free@plt>
  404c43:	48 8b 45 b0          	mov    -0x50(%rbp),%rax
  404c47:	48 89 c7             	mov    %rax,%rdi
  404c4a:	e8 21 cb ff ff       	callq  401770 <free@plt>
  404c4f:	83 45 e8 01          	addl   $0x1,-0x18(%rbp)
  404c53:	83 7d e8 01          	cmpl   $0x1,-0x18(%rbp)
  404c57:	0f 8e e1 fa ff ff    	jle    40473e <main+0x146>
  404c5d:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  404c61:	48 89 c7             	mov    %rax,%rdi
  404c64:	e8 07 cb ff ff       	callq  401770 <free@plt>
  404c69:	48 8d 85 00 fd ff ff 	lea    -0x300(%rbp),%rax
  404c70:	48 89 c7             	mov    %rax,%rdi
  404c73:	e8 44 06 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  404c78:	48 8d 9d a0 fc ff ff 	lea    -0x360(%rbp),%rbx
  404c7f:	48 83 c3 60          	add    $0x60,%rbx
  404c83:	48 8d 85 a0 fc ff ff 	lea    -0x360(%rbp),%rax
  404c8a:	48 39 c3             	cmp    %rax,%rbx
  404c8d:	74 0e                	je     404c9d <main+0x6a5>
  404c8f:	48 83 eb 60          	sub    $0x60,%rbx
  404c93:	48 89 df             	mov    %rbx,%rdi
  404c96:	e8 21 06 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  404c9b:	eb e6                	jmp    404c83 <main+0x68b>
  404c9d:	b8 00 00 00 00       	mov    $0x0,%eax
  404ca2:	e9 29 01 00 00       	jmpq   404dd0 <main+0x7d8>
  404ca7:	48 89 c3             	mov    %rax,%rbx
  404caa:	48 8d 85 d0 fe ff ff 	lea    -0x130(%rbp),%rax
  404cb1:	48 89 c7             	mov    %rax,%rdi
  404cb4:	e8 03 06 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  404cb9:	eb 03                	jmp    404cbe <main+0x6c6>
  404cbb:	48 89 c3             	mov    %rax,%rbx
  404cbe:	48 8d 85 30 ff ff ff 	lea    -0xd0(%rbp),%rax
  404cc5:	48 89 c7             	mov    %rax,%rdi
  404cc8:	e8 b3 cc ff ff       	callq  401980 <_ZNSsD1Ev@plt>
  404ccd:	eb 03                	jmp    404cd2 <main+0x6da>
  404ccf:	48 89 c3             	mov    %rax,%rbx
  404cd2:	48 8d 85 3f ff ff ff 	lea    -0xc1(%rbp),%rax
  404cd9:	48 89 c7             	mov    %rax,%rdi
  404cdc:	e8 cf cc ff ff       	callq  4019b0 <_ZNSaIcED1Ev@plt>
  404ce1:	e9 a3 00 00 00       	jmpq   404d89 <main+0x791>
  404ce6:	48 89 c3             	mov    %rax,%rbx
  404ce9:	48 8d 85 40 ff ff ff 	lea    -0xc0(%rbp),%rax
  404cf0:	48 89 c7             	mov    %rax,%rdi
  404cf3:	e8 88 cc ff ff       	callq  401980 <_ZNSsD1Ev@plt>
  404cf8:	eb 03                	jmp    404cfd <main+0x705>
  404cfa:	48 89 c3             	mov    %rax,%rbx
  404cfd:	48 8d 85 4f ff ff ff 	lea    -0xb1(%rbp),%rax
  404d04:	48 89 c7             	mov    %rax,%rdi
  404d07:	e8 a4 cc ff ff       	callq  4019b0 <_ZNSaIcED1Ev@plt>
  404d0c:	eb 67                	jmp    404d75 <main+0x77d>
  404d0e:	48 89 c3             	mov    %rax,%rbx
  404d11:	48 8d 85 50 ff ff ff 	lea    -0xb0(%rbp),%rax
  404d18:	48 89 c7             	mov    %rax,%rdi
  404d1b:	e8 60 cc ff ff       	callq  401980 <_ZNSsD1Ev@plt>
  404d20:	eb 53                	jmp    404d75 <main+0x77d>
  404d22:	48 89 c3             	mov    %rax,%rbx
  404d25:	48 8d 85 60 ff ff ff 	lea    -0xa0(%rbp),%rax
  404d2c:	48 89 c7             	mov    %rax,%rdi
  404d2f:	e8 4c cc ff ff       	callq  401980 <_ZNSsD1Ev@plt>
  404d34:	eb 03                	jmp    404d39 <main+0x741>
  404d36:	48 89 c3             	mov    %rax,%rbx
  404d39:	48 8d 85 6f ff ff ff 	lea    -0x91(%rbp),%rax
  404d40:	48 89 c7             	mov    %rax,%rdi
  404d43:	e8 68 cc ff ff       	callq  4019b0 <_ZNSaIcED1Ev@plt>
  404d48:	eb 17                	jmp    404d61 <main+0x769>
  404d4a:	48 89 c3             	mov    %rax,%rbx
  404d4d:	48 8d 85 70 ff ff ff 	lea    -0x90(%rbp),%rax
  404d54:	48 89 c7             	mov    %rax,%rdi
  404d57:	e8 24 cc ff ff       	callq  401980 <_ZNSsD1Ev@plt>
  404d5c:	eb 03                	jmp    404d61 <main+0x769>
  404d5e:	48 89 c3             	mov    %rax,%rbx
  404d61:	48 8d 85 c0 fe ff ff 	lea    -0x140(%rbp),%rax
  404d68:	48 89 c7             	mov    %rax,%rdi
  404d6b:	e8 10 cc ff ff       	callq  401980 <_ZNSsD1Ev@plt>
  404d70:	eb 03                	jmp    404d75 <main+0x77d>
  404d72:	48 89 c3             	mov    %rax,%rbx
  404d75:	48 8d 85 60 fd ff ff 	lea    -0x2a0(%rbp),%rax
  404d7c:	48 89 c7             	mov    %rax,%rdi
  404d7f:	e8 cc c9 ff ff       	callq  401750 <_ZNSt19basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev@plt>
  404d84:	eb 03                	jmp    404d89 <main+0x791>
  404d86:	48 89 c3             	mov    %rax,%rbx
  404d89:	48 8d 85 00 fd ff ff 	lea    -0x300(%rbp),%rax
  404d90:	48 89 c7             	mov    %rax,%rdi
  404d93:	e8 24 05 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  404d98:	49 89 dc             	mov    %rbx,%r12
  404d9b:	eb 03                	jmp    404da0 <main+0x7a8>
  404d9d:	49 89 c4             	mov    %rax,%r12
  404da0:	48 8d 9d a0 fc ff ff 	lea    -0x360(%rbp),%rbx
  404da7:	48 83 c3 60          	add    $0x60,%rbx
  404dab:	48 8d 85 a0 fc ff ff 	lea    -0x360(%rbp),%rax
  404db2:	48 39 c3             	cmp    %rax,%rbx
  404db5:	74 0e                	je     404dc5 <main+0x7cd>
  404db7:	48 83 eb 60          	sub    $0x60,%rbx
  404dbb:	48 89 df             	mov    %rbx,%rdi
  404dbe:	e8 f9 04 00 00       	callq  4052bc <_ZN2cv3MatD1Ev>
  404dc3:	eb e6                	jmp    404dab <main+0x7b3>
  404dc5:	4c 89 e0             	mov    %r12,%rax
  404dc8:	48 89 c7             	mov    %rax,%rdi
  404dcb:	e8 70 ca ff ff       	callq  401840 <_Unwind_Resume@plt>
  404dd0:	48 81 c4 90 03 00 00 	add    $0x390,%rsp
  404dd7:	5b                   	pop    %rbx
  404dd8:	41 5c                	pop    %r12
  404dda:	5d                   	pop    %rbp
  404ddb:	c3                   	retq   

0000000000404ddc <_Z41__static_initialization_and_destruction_0ii>:
  404ddc:	55                   	push   %rbp
  404ddd:	48 89 e5             	mov    %rsp,%rbp
  404de0:	48 83 ec 10          	sub    $0x10,%rsp
  404de4:	89 7d fc             	mov    %edi,-0x4(%rbp)
  404de7:	89 75 f8             	mov    %esi,-0x8(%rbp)
  404dea:	83 7d fc 01          	cmpl   $0x1,-0x4(%rbp)
  404dee:	75 27                	jne    404e17 <_Z41__static_initialization_and_destruction_0ii+0x3b>
  404df0:	81 7d f8 ff ff 00 00 	cmpl   $0xffff,-0x8(%rbp)
  404df7:	75 1e                	jne    404e17 <_Z41__static_initialization_and_destruction_0ii+0x3b>
  404df9:	bf d1 92 60 00       	mov    $0x6092d1,%edi
  404dfe:	e8 2d cb ff ff       	callq  401930 <_ZNSt8ios_base4InitC1Ev@plt>
  404e03:	ba 08 70 40 00       	mov    $0x407008,%edx
  404e08:	be d1 92 60 00       	mov    $0x6092d1,%esi
  404e0d:	bf 50 18 40 00       	mov    $0x401850,%edi
  404e12:	e8 b9 ca ff ff       	callq  4018d0 <__cxa_atexit@plt>
  404e17:	c9                   	leaveq 
  404e18:	c3                   	retq   

0000000000404e19 <_GLOBAL__sub_I_home>:
  404e19:	55                   	push   %rbp
  404e1a:	48 89 e5             	mov    %rsp,%rbp
  404e1d:	be ff ff 00 00       	mov    $0xffff,%esi
  404e22:	bf 01 00 00 00       	mov    $0x1,%edi
  404e27:	e8 b0 ff ff ff       	callq  404ddc <_Z41__static_initialization_and_destruction_0ii>
  404e2c:	5d                   	pop    %rbp
  404e2d:	c3                   	retq   

0000000000404e2e <_ZnwmPv>:
  404e2e:	55                   	push   %rbp
  404e2f:	48 89 e5             	mov    %rsp,%rbp
  404e32:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  404e36:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  404e3a:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  404e3e:	5d                   	pop    %rbp
  404e3f:	c3                   	retq   

0000000000404e40 <_ZdlPvS_>:
  404e40:	55                   	push   %rbp
  404e41:	48 89 e5             	mov    %rsp,%rbp
  404e44:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  404e48:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  404e4c:	5d                   	pop    %rbp
  404e4d:	c3                   	retq   

0000000000404e4e <_ZN2cv5RangeC1Eii>:
  404e4e:	55                   	push   %rbp
  404e4f:	48 89 e5             	mov    %rsp,%rbp
  404e52:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  404e56:	89 75 f4             	mov    %esi,-0xc(%rbp)
  404e59:	89 55 f0             	mov    %edx,-0x10(%rbp)
  404e5c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404e60:	8b 55 f4             	mov    -0xc(%rbp),%edx
  404e63:	89 10                	mov    %edx,(%rax)
  404e65:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404e69:	8b 55 f0             	mov    -0x10(%rbp),%edx
  404e6c:	89 50 04             	mov    %edx,0x4(%rax)
  404e6f:	5d                   	pop    %rbp
  404e70:	c3                   	retq   
  404e71:	90                   	nop

0000000000404e72 <_ZN2cv3Mat9initEmptyEv>:
  404e72:	55                   	push   %rbp
  404e73:	48 89 e5             	mov    %rsp,%rbp
  404e76:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  404e7a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404e7e:	c7 00 00 00 ff 42    	movl   $0x42ff0000,(%rax)
  404e84:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404e88:	c7 40 0c 00 00 00 00 	movl   $0x0,0xc(%rax)
  404e8f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404e93:	8b 50 0c             	mov    0xc(%rax),%edx
  404e96:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404e9a:	89 50 08             	mov    %edx,0x8(%rax)
  404e9d:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404ea1:	8b 50 08             	mov    0x8(%rax),%edx
  404ea4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404ea8:	89 50 04             	mov    %edx,0x4(%rax)
  404eab:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404eaf:	48 c7 40 30 00 00 00 	movq   $0x0,0x30(%rax)
  404eb6:	00 
  404eb7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404ebb:	48 8b 50 30          	mov    0x30(%rax),%rdx
  404ebf:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404ec3:	48 89 50 28          	mov    %rdx,0x28(%rax)
  404ec7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404ecb:	48 8b 50 28          	mov    0x28(%rax),%rdx
  404ecf:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404ed3:	48 89 50 20          	mov    %rdx,0x20(%rax)
  404ed7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404edb:	48 8b 50 20          	mov    0x20(%rax),%rdx
  404edf:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404ee3:	48 89 50 10          	mov    %rdx,0x10(%rax)
  404ee7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404eeb:	48 c7 40 18 00 00 00 	movq   $0x0,0x18(%rax)
  404ef2:	00 
  404ef3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404ef7:	48 c7 40 38 00 00 00 	movq   $0x0,0x38(%rax)
  404efe:	00 
  404eff:	5d                   	pop    %rbp
  404f00:	c3                   	retq   
  404f01:	90                   	nop

0000000000404f02 <_ZN2cv3MatC1Ev>:
  404f02:	55                   	push   %rbp
  404f03:	48 89 e5             	mov    %rsp,%rbp
  404f06:	48 83 ec 10          	sub    $0x10,%rsp
  404f0a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  404f0e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404f12:	48 8d 50 08          	lea    0x8(%rax),%rdx
  404f16:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404f1a:	48 83 c0 40          	add    $0x40,%rax
  404f1e:	48 89 d6             	mov    %rdx,%rsi
  404f21:	48 89 c7             	mov    %rax,%rdi
  404f24:	e8 2d 06 00 00       	callq  405556 <_ZN2cv3Mat5MSizeC1EPi>
  404f29:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404f2d:	48 83 c0 48          	add    $0x48,%rax
  404f31:	48 89 c7             	mov    %rax,%rdi
  404f34:	e8 37 06 00 00       	callq  405570 <_ZN2cv3Mat5MStepC1Ev>
  404f39:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  404f3d:	48 89 c7             	mov    %rax,%rdi
  404f40:	e8 2d ff ff ff       	callq  404e72 <_ZN2cv3Mat9initEmptyEv>
  404f45:	c9                   	leaveq 
  404f46:	c3                   	retq   
  404f47:	90                   	nop

0000000000404f48 <_ZN2cv3MatC1ERKS0_>:
  404f48:	55                   	push   %rbp
  404f49:	48 89 e5             	mov    %rsp,%rbp
  404f4c:	53                   	push   %rbx
  404f4d:	48 83 ec 18          	sub    $0x18,%rsp
  404f51:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  404f55:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  404f59:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  404f5d:	8b 10                	mov    (%rax),%edx
  404f5f:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  404f63:	89 10                	mov    %edx,(%rax)
  404f65:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  404f69:	8b 50 04             	mov    0x4(%rax),%edx
  404f6c:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  404f70:	89 50 04             	mov    %edx,0x4(%rax)
  404f73:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  404f77:	8b 50 08             	mov    0x8(%rax),%edx
  404f7a:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  404f7e:	89 50 08             	mov    %edx,0x8(%rax)
  404f81:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  404f85:	8b 50 0c             	mov    0xc(%rax),%edx
  404f88:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  404f8c:	89 50 0c             	mov    %edx,0xc(%rax)
  404f8f:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  404f93:	48 8b 50 10          	mov    0x10(%rax),%rdx
  404f97:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  404f9b:	48 89 50 10          	mov    %rdx,0x10(%rax)
  404f9f:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  404fa3:	48 8b 50 18          	mov    0x18(%rax),%rdx
  404fa7:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  404fab:	48 89 50 18          	mov    %rdx,0x18(%rax)
  404faf:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  404fb3:	48 8b 50 20          	mov    0x20(%rax),%rdx
  404fb7:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  404fbb:	48 89 50 20          	mov    %rdx,0x20(%rax)
  404fbf:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  404fc3:	48 8b 50 28          	mov    0x28(%rax),%rdx
  404fc7:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  404fcb:	48 89 50 28          	mov    %rdx,0x28(%rax)
  404fcf:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  404fd3:	48 8b 50 30          	mov    0x30(%rax),%rdx
  404fd7:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  404fdb:	48 89 50 30          	mov    %rdx,0x30(%rax)
  404fdf:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  404fe3:	48 8b 50 38          	mov    0x38(%rax),%rdx
  404fe7:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  404feb:	48 89 50 38          	mov    %rdx,0x38(%rax)
  404fef:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  404ff3:	48 8d 50 08          	lea    0x8(%rax),%rdx
  404ff7:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  404ffb:	48 83 c0 40          	add    $0x40,%rax
  404fff:	48 89 d6             	mov    %rdx,%rsi
  405002:	48 89 c7             	mov    %rax,%rdi
  405005:	e8 4c 05 00 00       	callq  405556 <_ZN2cv3Mat5MSizeC1EPi>
  40500a:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40500e:	48 83 c0 48          	add    $0x48,%rax
  405012:	48 89 c7             	mov    %rax,%rdi
  405015:	e8 56 05 00 00       	callq  405570 <_ZN2cv3Mat5MStepC1Ev>
  40501a:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40501e:	48 8b 40 18          	mov    0x18(%rax),%rax
  405022:	48 85 c0             	test   %rax,%rax
  405025:	74 0c                	je     405033 <_ZN2cv3MatC1ERKS0_+0xeb>
  405027:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40502b:	48 8b 40 18          	mov    0x18(%rax),%rax
  40502f:	f0 83 00 01          	lock addl $0x1,(%rax)
  405033:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  405037:	8b 40 04             	mov    0x4(%rax),%eax
  40503a:	83 f8 02             	cmp    $0x2,%eax
  40503d:	7f 68                	jg     4050a7 <_ZN2cv3MatC1ERKS0_+0x15f>
  40503f:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405043:	48 83 c0 48          	add    $0x48,%rax
  405047:	be 00 00 00 00       	mov    $0x0,%esi
  40504c:	48 89 c7             	mov    %rax,%rdi
  40504f:	e8 76 05 00 00       	callq  4055ca <_ZN2cv3Mat5MStepixEi>
  405054:	48 89 c3             	mov    %rax,%rbx
  405057:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40505b:	48 83 c0 48          	add    $0x48,%rax
  40505f:	be 00 00 00 00       	mov    $0x0,%esi
  405064:	48 89 c7             	mov    %rax,%rdi
  405067:	e8 3c 05 00 00       	callq  4055a8 <_ZNK2cv3Mat5MStepixEi>
  40506c:	48 8b 00             	mov    (%rax),%rax
  40506f:	48 89 03             	mov    %rax,(%rbx)
  405072:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405076:	48 83 c0 48          	add    $0x48,%rax
  40507a:	be 01 00 00 00       	mov    $0x1,%esi
  40507f:	48 89 c7             	mov    %rax,%rdi
  405082:	e8 43 05 00 00       	callq  4055ca <_ZN2cv3Mat5MStepixEi>
  405087:	48 89 c3             	mov    %rax,%rbx
  40508a:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40508e:	48 83 c0 48          	add    $0x48,%rax
  405092:	be 01 00 00 00       	mov    $0x1,%esi
  405097:	48 89 c7             	mov    %rax,%rdi
  40509a:	e8 09 05 00 00       	callq  4055a8 <_ZNK2cv3Mat5MStepixEi>
  40509f:	48 8b 00             	mov    (%rax),%rax
  4050a2:	48 89 03             	mov    %rax,(%rbx)
  4050a5:	eb 1e                	jmp    4050c5 <_ZN2cv3MatC1ERKS0_+0x17d>
  4050a7:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4050ab:	c7 40 04 00 00 00 00 	movl   $0x0,0x4(%rax)
  4050b2:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
  4050b6:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4050ba:	48 89 d6             	mov    %rdx,%rsi
  4050bd:	48 89 c7             	mov    %rax,%rdi
  4050c0:	e8 6b c6 ff ff       	callq  401730 <_ZN2cv3Mat8copySizeERKS0_@plt>
  4050c5:	48 83 c4 18          	add    $0x18,%rsp
  4050c9:	5b                   	pop    %rbx
  4050ca:	5d                   	pop    %rbp
  4050cb:	c3                   	retq   

00000000004050cc <_ZN2cv3MatC1EiiiPvm>:
  4050cc:	55                   	push   %rbp
  4050cd:	48 89 e5             	mov    %rsp,%rbp
  4050d0:	48 83 ec 40          	sub    $0x40,%rsp
  4050d4:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4050d8:	89 75 e4             	mov    %esi,-0x1c(%rbp)
  4050db:	89 55 e0             	mov    %edx,-0x20(%rbp)
  4050de:	89 4d dc             	mov    %ecx,-0x24(%rbp)
  4050e1:	4c 89 45 d0          	mov    %r8,-0x30(%rbp)
  4050e5:	4c 89 4d c8          	mov    %r9,-0x38(%rbp)
  4050e9:	8b 45 dc             	mov    -0x24(%rbp),%eax
  4050ec:	25 ff 0f 00 00       	and    $0xfff,%eax
  4050f1:	05 00 00 ff 42       	add    $0x42ff0000,%eax
  4050f6:	89 c2                	mov    %eax,%edx
  4050f8:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4050fc:	89 10                	mov    %edx,(%rax)
  4050fe:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405102:	c7 40 04 02 00 00 00 	movl   $0x2,0x4(%rax)
  405109:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40510d:	8b 55 e4             	mov    -0x1c(%rbp),%edx
  405110:	89 50 08             	mov    %edx,0x8(%rax)
  405113:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405117:	8b 55 e0             	mov    -0x20(%rbp),%edx
  40511a:	89 50 0c             	mov    %edx,0xc(%rax)
  40511d:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405121:	48 8b 55 d0          	mov    -0x30(%rbp),%rdx
  405125:	48 89 50 10          	mov    %rdx,0x10(%rax)
  405129:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40512d:	48 c7 40 18 00 00 00 	movq   $0x0,0x18(%rax)
  405134:	00 
  405135:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405139:	48 8b 55 d0          	mov    -0x30(%rbp),%rdx
  40513d:	48 89 50 20          	mov    %rdx,0x20(%rax)
  405141:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405145:	48 c7 40 28 00 00 00 	movq   $0x0,0x28(%rax)
  40514c:	00 
  40514d:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405151:	48 c7 40 30 00 00 00 	movq   $0x0,0x30(%rax)
  405158:	00 
  405159:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40515d:	48 c7 40 38 00 00 00 	movq   $0x0,0x38(%rax)
  405164:	00 
  405165:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405169:	48 8d 50 08          	lea    0x8(%rax),%rdx
  40516d:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405171:	48 83 c0 40          	add    $0x40,%rax
  405175:	48 89 d6             	mov    %rdx,%rsi
  405178:	48 89 c7             	mov    %rax,%rdi
  40517b:	e8 d6 03 00 00       	callq  405556 <_ZN2cv3Mat5MSizeC1EPi>
  405180:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405184:	48 83 c0 48          	add    $0x48,%rax
  405188:	48 89 c7             	mov    %rax,%rdi
  40518b:	e8 e0 03 00 00       	callq  405570 <_ZN2cv3Mat5MStepC1Ev>
  405190:	8b 45 dc             	mov    -0x24(%rbp),%eax
  405193:	25 f8 0f 00 00       	and    $0xff8,%eax
  405198:	c1 f8 03             	sar    $0x3,%eax
  40519b:	8d 50 01             	lea    0x1(%rax),%edx
  40519e:	b8 02 00 00 00       	mov    $0x2,%eax
  4051a3:	48 83 c0 01          	add    $0x1,%rax
  4051a7:	48 c1 e0 0e          	shl    $0xe,%rax
  4051ab:	48 0d 50 3a 00 00    	or     $0x3a50,%rax
  4051b1:	48 89 c6             	mov    %rax,%rsi
  4051b4:	8b 45 dc             	mov    -0x24(%rbp),%eax
  4051b7:	83 e0 07             	and    $0x7,%eax
  4051ba:	01 c0                	add    %eax,%eax
  4051bc:	89 c1                	mov    %eax,%ecx
  4051be:	48 d3 ee             	shr    %cl,%rsi
  4051c1:	48 89 f0             	mov    %rsi,%rax
  4051c4:	83 e0 03             	and    $0x3,%eax
  4051c7:	89 c1                	mov    %eax,%ecx
  4051c9:	d3 e2                	shl    %cl,%edx
  4051cb:	89 d0                	mov    %edx,%eax
  4051cd:	48 98                	cltq   
  4051cf:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4051d3:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4051d7:	8b 40 0c             	mov    0xc(%rax),%eax
  4051da:	48 98                	cltq   
  4051dc:	48 0f af 45 f8       	imul   -0x8(%rbp),%rax
  4051e1:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  4051e5:	48 83 7d c8 00       	cmpq   $0x0,-0x38(%rbp)
  4051ea:	75 1b                	jne    405207 <_ZN2cv3MatC1EiiiPvm+0x13b>
  4051ec:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4051f0:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
  4051f4:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4051f8:	8b 00                	mov    (%rax),%eax
  4051fa:	80 cc 40             	or     $0x40,%ah
  4051fd:	89 c2                	mov    %eax,%edx
  4051ff:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405203:	89 10                	mov    %edx,(%rax)
  405205:	eb 38                	jmp    40523f <_ZN2cv3MatC1EiiiPvm+0x173>
  405207:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40520b:	8b 40 08             	mov    0x8(%rax),%eax
  40520e:	83 f8 01             	cmp    $0x1,%eax
  405211:	75 08                	jne    40521b <_ZN2cv3MatC1EiiiPvm+0x14f>
  405213:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  405217:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
  40521b:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40521f:	8b 10                	mov    (%rax),%edx
  405221:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  405225:	48 3b 45 f0          	cmp    -0x10(%rbp),%rax
  405229:	75 07                	jne    405232 <_ZN2cv3MatC1EiiiPvm+0x166>
  40522b:	b8 00 40 00 00       	mov    $0x4000,%eax
  405230:	eb 05                	jmp    405237 <_ZN2cv3MatC1EiiiPvm+0x16b>
  405232:	b8 00 00 00 00       	mov    $0x0,%eax
  405237:	09 c2                	or     %eax,%edx
  405239:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40523d:	89 10                	mov    %edx,(%rax)
  40523f:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405243:	48 83 c0 48          	add    $0x48,%rax
  405247:	be 00 00 00 00       	mov    $0x0,%esi
  40524c:	48 89 c7             	mov    %rax,%rdi
  40524f:	e8 76 03 00 00       	callq  4055ca <_ZN2cv3Mat5MStepixEi>
  405254:	48 8b 55 c8          	mov    -0x38(%rbp),%rdx
  405258:	48 89 10             	mov    %rdx,(%rax)
  40525b:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40525f:	48 83 c0 48          	add    $0x48,%rax
  405263:	be 01 00 00 00       	mov    $0x1,%esi
  405268:	48 89 c7             	mov    %rax,%rdi
  40526b:	e8 5a 03 00 00       	callq  4055ca <_ZN2cv3Mat5MStepixEi>
  405270:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  405274:	48 89 10             	mov    %rdx,(%rax)
  405277:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40527b:	48 8b 50 20          	mov    0x20(%rax),%rdx
  40527f:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405283:	8b 40 08             	mov    0x8(%rax),%eax
  405286:	48 98                	cltq   
  405288:	48 0f af 45 c8       	imul   -0x38(%rbp),%rax
  40528d:	48 01 c2             	add    %rax,%rdx
  405290:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405294:	48 89 50 30          	mov    %rdx,0x30(%rax)
  405298:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40529c:	48 8b 40 30          	mov    0x30(%rax),%rax
  4052a0:	48 8b 55 c8          	mov    -0x38(%rbp),%rdx
  4052a4:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  4052a8:	48 29 d1             	sub    %rdx,%rcx
  4052ab:	48 89 ca             	mov    %rcx,%rdx
  4052ae:	48 01 c2             	add    %rax,%rdx
  4052b1:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4052b5:	48 89 50 28          	mov    %rdx,0x28(%rax)
  4052b9:	c9                   	leaveq 
  4052ba:	c3                   	retq   
  4052bb:	90                   	nop

00000000004052bc <_ZN2cv3MatD1Ev>:
  4052bc:	55                   	push   %rbp
  4052bd:	48 89 e5             	mov    %rsp,%rbp
  4052c0:	48 83 ec 10          	sub    $0x10,%rsp
  4052c4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4052c8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4052cc:	48 89 c7             	mov    %rax,%rdi
  4052cf:	e8 a8 01 00 00       	callq  40547c <_ZN2cv3Mat7releaseEv>
  4052d4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4052d8:	48 8b 40 48          	mov    0x48(%rax),%rax
  4052dc:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  4052e0:	48 83 c2 50          	add    $0x50,%rdx
  4052e4:	48 39 d0             	cmp    %rdx,%rax
  4052e7:	74 10                	je     4052f9 <_ZN2cv3MatD1Ev+0x3d>
  4052e9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4052ed:	48 8b 40 48          	mov    0x48(%rax),%rax
  4052f1:	48 89 c7             	mov    %rax,%rdi
  4052f4:	e8 e7 c5 ff ff       	callq  4018e0 <_ZN2cv8fastFreeEPv@plt>
  4052f9:	c9                   	leaveq 
  4052fa:	c3                   	retq   
  4052fb:	90                   	nop

00000000004052fc <_ZN2cv3MataSERKS0_>:
  4052fc:	55                   	push   %rbp
  4052fd:	48 89 e5             	mov    %rsp,%rbp
  405300:	53                   	push   %rbx
  405301:	48 83 ec 18          	sub    $0x18,%rsp
  405305:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  405309:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  40530d:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405311:	48 3b 45 e0          	cmp    -0x20(%rbp),%rax
  405315:	0f 84 56 01 00 00    	je     405471 <_ZN2cv3MataSERKS0_+0x175>
  40531b:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40531f:	48 8b 40 18          	mov    0x18(%rax),%rax
  405323:	48 85 c0             	test   %rax,%rax
  405326:	74 0c                	je     405334 <_ZN2cv3MataSERKS0_+0x38>
  405328:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40532c:	48 8b 40 18          	mov    0x18(%rax),%rax
  405330:	f0 83 00 01          	lock addl $0x1,(%rax)
  405334:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405338:	48 89 c7             	mov    %rax,%rdi
  40533b:	e8 3c 01 00 00       	callq  40547c <_ZN2cv3Mat7releaseEv>
  405340:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  405344:	8b 10                	mov    (%rax),%edx
  405346:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40534a:	89 10                	mov    %edx,(%rax)
  40534c:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405350:	8b 40 04             	mov    0x4(%rax),%eax
  405353:	83 f8 02             	cmp    $0x2,%eax
  405356:	0f 8f a2 00 00 00    	jg     4053fe <_ZN2cv3MataSERKS0_+0x102>
  40535c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  405360:	8b 40 04             	mov    0x4(%rax),%eax
  405363:	83 f8 02             	cmp    $0x2,%eax
  405366:	0f 8f 92 00 00 00    	jg     4053fe <_ZN2cv3MataSERKS0_+0x102>
  40536c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  405370:	8b 50 04             	mov    0x4(%rax),%edx
  405373:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405377:	89 50 04             	mov    %edx,0x4(%rax)
  40537a:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40537e:	8b 50 08             	mov    0x8(%rax),%edx
  405381:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405385:	89 50 08             	mov    %edx,0x8(%rax)
  405388:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40538c:	8b 50 0c             	mov    0xc(%rax),%edx
  40538f:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405393:	89 50 0c             	mov    %edx,0xc(%rax)
  405396:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40539a:	48 83 c0 48          	add    $0x48,%rax
  40539e:	be 00 00 00 00       	mov    $0x0,%esi
  4053a3:	48 89 c7             	mov    %rax,%rdi
  4053a6:	e8 1f 02 00 00       	callq  4055ca <_ZN2cv3Mat5MStepixEi>
  4053ab:	48 89 c3             	mov    %rax,%rbx
  4053ae:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4053b2:	48 83 c0 48          	add    $0x48,%rax
  4053b6:	be 00 00 00 00       	mov    $0x0,%esi
  4053bb:	48 89 c7             	mov    %rax,%rdi
  4053be:	e8 e5 01 00 00       	callq  4055a8 <_ZNK2cv3Mat5MStepixEi>
  4053c3:	48 8b 00             	mov    (%rax),%rax
  4053c6:	48 89 03             	mov    %rax,(%rbx)
  4053c9:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4053cd:	48 83 c0 48          	add    $0x48,%rax
  4053d1:	be 01 00 00 00       	mov    $0x1,%esi
  4053d6:	48 89 c7             	mov    %rax,%rdi
  4053d9:	e8 ec 01 00 00       	callq  4055ca <_ZN2cv3Mat5MStepixEi>
  4053de:	48 89 c3             	mov    %rax,%rbx
  4053e1:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4053e5:	48 83 c0 48          	add    $0x48,%rax
  4053e9:	be 01 00 00 00       	mov    $0x1,%esi
  4053ee:	48 89 c7             	mov    %rax,%rdi
  4053f1:	e8 b2 01 00 00       	callq  4055a8 <_ZNK2cv3Mat5MStepixEi>
  4053f6:	48 8b 00             	mov    (%rax),%rax
  4053f9:	48 89 03             	mov    %rax,(%rbx)
  4053fc:	eb 13                	jmp    405411 <_ZN2cv3MataSERKS0_+0x115>
  4053fe:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
  405402:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405406:	48 89 d6             	mov    %rdx,%rsi
  405409:	48 89 c7             	mov    %rax,%rdi
  40540c:	e8 1f c3 ff ff       	callq  401730 <_ZN2cv3Mat8copySizeERKS0_@plt>
  405411:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  405415:	48 8b 50 10          	mov    0x10(%rax),%rdx
  405419:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40541d:	48 89 50 10          	mov    %rdx,0x10(%rax)
  405421:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  405425:	48 8b 50 20          	mov    0x20(%rax),%rdx
  405429:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40542d:	48 89 50 20          	mov    %rdx,0x20(%rax)
  405431:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  405435:	48 8b 50 28          	mov    0x28(%rax),%rdx
  405439:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40543d:	48 89 50 28          	mov    %rdx,0x28(%rax)
  405441:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  405445:	48 8b 50 30          	mov    0x30(%rax),%rdx
  405449:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40544d:	48 89 50 30          	mov    %rdx,0x30(%rax)
  405451:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  405455:	48 8b 50 18          	mov    0x18(%rax),%rdx
  405459:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40545d:	48 89 50 18          	mov    %rdx,0x18(%rax)
  405461:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  405465:	48 8b 50 38          	mov    0x38(%rax),%rdx
  405469:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40546d:	48 89 50 38          	mov    %rdx,0x38(%rax)
  405471:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405475:	48 83 c4 18          	add    $0x18,%rsp
  405479:	5b                   	pop    %rbx
  40547a:	5d                   	pop    %rbp
  40547b:	c3                   	retq   

000000000040547c <_ZN2cv3Mat7releaseEv>:
  40547c:	55                   	push   %rbp
  40547d:	48 89 e5             	mov    %rsp,%rbp
  405480:	48 83 ec 10          	sub    $0x10,%rsp
  405484:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  405488:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40548c:	48 8b 40 18          	mov    0x18(%rax),%rax
  405490:	48 85 c0             	test   %rax,%rax
  405493:	74 1d                	je     4054b2 <_ZN2cv3Mat7releaseEv+0x36>
  405495:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405499:	48 8b 40 18          	mov    0x18(%rax),%rax
  40549d:	ba ff ff ff ff       	mov    $0xffffffff,%edx
  4054a2:	f0 0f c1 10          	lock xadd %edx,(%rax)
  4054a6:	83 fa 01             	cmp    $0x1,%edx
  4054a9:	75 07                	jne    4054b2 <_ZN2cv3Mat7releaseEv+0x36>
  4054ab:	b8 01 00 00 00       	mov    $0x1,%eax
  4054b0:	eb 05                	jmp    4054b7 <_ZN2cv3Mat7releaseEv+0x3b>
  4054b2:	b8 00 00 00 00       	mov    $0x0,%eax
  4054b7:	84 c0                	test   %al,%al
  4054b9:	74 0c                	je     4054c7 <_ZN2cv3Mat7releaseEv+0x4b>
  4054bb:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4054bf:	48 89 c7             	mov    %rax,%rdi
  4054c2:	e8 f9 c2 ff ff       	callq  4017c0 <_ZN2cv3Mat10deallocateEv@plt>
  4054c7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4054cb:	48 c7 40 30 00 00 00 	movq   $0x0,0x30(%rax)
  4054d2:	00 
  4054d3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4054d7:	48 8b 50 30          	mov    0x30(%rax),%rdx
  4054db:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4054df:	48 89 50 28          	mov    %rdx,0x28(%rax)
  4054e3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4054e7:	48 8b 50 28          	mov    0x28(%rax),%rdx
  4054eb:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4054ef:	48 89 50 20          	mov    %rdx,0x20(%rax)
  4054f3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4054f7:	48 8b 50 20          	mov    0x20(%rax),%rdx
  4054fb:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4054ff:	48 89 50 10          	mov    %rdx,0x10(%rax)
  405503:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405507:	48 8b 40 40          	mov    0x40(%rax),%rax
  40550b:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  405511:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405515:	48 c7 40 18 00 00 00 	movq   $0x0,0x18(%rax)
  40551c:	00 
  40551d:	c9                   	leaveq 
  40551e:	c3                   	retq   
  40551f:	90                   	nop

0000000000405520 <_ZNK2cv3MatclENS_5RangeES1_>:
  405520:	55                   	push   %rbp
  405521:	48 89 e5             	mov    %rsp,%rbp
  405524:	48 83 ec 30          	sub    $0x30,%rsp
  405528:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40552c:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  405530:	48 89 55 e0          	mov    %rdx,-0x20(%rbp)
  405534:	48 89 4d d0          	mov    %rcx,-0x30(%rbp)
  405538:	48 8d 4d d0          	lea    -0x30(%rbp),%rcx
  40553c:	48 8d 55 e0          	lea    -0x20(%rbp),%rdx
  405540:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  405544:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405548:	48 89 c7             	mov    %rax,%rdi
  40554b:	e8 20 c3 ff ff       	callq  401870 <_ZN2cv3MatC1ERKS0_RKNS_5RangeES5_@plt>
  405550:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405554:	c9                   	leaveq 
  405555:	c3                   	retq   

0000000000405556 <_ZN2cv3Mat5MSizeC1EPi>:
  405556:	55                   	push   %rbp
  405557:	48 89 e5             	mov    %rsp,%rbp
  40555a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40555e:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  405562:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405566:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  40556a:	48 89 10             	mov    %rdx,(%rax)
  40556d:	5d                   	pop    %rbp
  40556e:	c3                   	retq   
  40556f:	90                   	nop

0000000000405570 <_ZN2cv3Mat5MStepC1Ev>:
  405570:	55                   	push   %rbp
  405571:	48 89 e5             	mov    %rsp,%rbp
  405574:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  405578:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40557c:	48 8d 50 08          	lea    0x8(%rax),%rdx
  405580:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405584:	48 89 10             	mov    %rdx,(%rax)
  405587:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40558b:	48 8b 10             	mov    (%rax),%rdx
  40558e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405592:	48 8b 00             	mov    (%rax),%rax
  405595:	48 83 c0 08          	add    $0x8,%rax
  405599:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  4055a0:	48 8b 00             	mov    (%rax),%rax
  4055a3:	48 89 02             	mov    %rax,(%rdx)
  4055a6:	5d                   	pop    %rbp
  4055a7:	c3                   	retq   

00000000004055a8 <_ZNK2cv3Mat5MStepixEi>:
  4055a8:	55                   	push   %rbp
  4055a9:	48 89 e5             	mov    %rsp,%rbp
  4055ac:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4055b0:	89 75 f4             	mov    %esi,-0xc(%rbp)
  4055b3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4055b7:	48 8b 00             	mov    (%rax),%rax
  4055ba:	8b 55 f4             	mov    -0xc(%rbp),%edx
  4055bd:	48 63 d2             	movslq %edx,%rdx
  4055c0:	48 c1 e2 03          	shl    $0x3,%rdx
  4055c4:	48 01 d0             	add    %rdx,%rax
  4055c7:	5d                   	pop    %rbp
  4055c8:	c3                   	retq   
  4055c9:	90                   	nop

00000000004055ca <_ZN2cv3Mat5MStepixEi>:
  4055ca:	55                   	push   %rbp
  4055cb:	48 89 e5             	mov    %rsp,%rbp
  4055ce:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4055d2:	89 75 f4             	mov    %esi,-0xc(%rbp)
  4055d5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4055d9:	48 8b 00             	mov    (%rax),%rax
  4055dc:	8b 55 f4             	mov    -0xc(%rbp),%edx
  4055df:	48 63 d2             	movslq %edx,%rdx
  4055e2:	48 c1 e2 03          	shl    $0x3,%rdx
  4055e6:	48 01 d0             	add    %rdx,%rax
  4055e9:	5d                   	pop    %rbp
  4055ea:	c3                   	retq   
  4055eb:	90                   	nop

00000000004055ec <_ZNSt6vectorIiSaIiEEC1Ev>:
  4055ec:	55                   	push   %rbp
  4055ed:	48 89 e5             	mov    %rsp,%rbp
  4055f0:	48 83 ec 10          	sub    $0x10,%rsp
  4055f4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4055f8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4055fc:	48 89 c7             	mov    %rax,%rdi
  4055ff:	e8 4e 02 00 00       	callq  405852 <_ZNSt12_Vector_baseIiSaIiEEC1Ev>
  405604:	c9                   	leaveq 
  405605:	c3                   	retq   

0000000000405606 <_ZNSt6vectorIiSaIiEED1Ev>:
  405606:	55                   	push   %rbp
  405607:	48 89 e5             	mov    %rsp,%rbp
  40560a:	53                   	push   %rbx
  40560b:	48 83 ec 18          	sub    $0x18,%rsp
  40560f:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  405613:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405617:	48 89 c7             	mov    %rax,%rdi
  40561a:	e8 bf 02 00 00       	callq  4058de <_ZNSt12_Vector_baseIiSaIiEE19_M_get_Tp_allocatorEv>
  40561f:	48 89 c2             	mov    %rax,%rdx
  405622:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405626:	48 8b 48 08          	mov    0x8(%rax),%rcx
  40562a:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40562e:	48 8b 00             	mov    (%rax),%rax
  405631:	48 89 ce             	mov    %rcx,%rsi
  405634:	48 89 c7             	mov    %rax,%rdi
  405637:	e8 b0 02 00 00       	callq  4058ec <_ZSt8_DestroyIPiiEvT_S1_RSaIT0_E>
  40563c:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405640:	48 89 c7             	mov    %rax,%rdi
  405643:	e8 24 02 00 00       	callq  40586c <_ZNSt12_Vector_baseIiSaIiEED1Ev>
  405648:	eb 1a                	jmp    405664 <_ZNSt6vectorIiSaIiEED1Ev+0x5e>
  40564a:	48 89 c3             	mov    %rax,%rbx
  40564d:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405651:	48 89 c7             	mov    %rax,%rdi
  405654:	e8 13 02 00 00       	callq  40586c <_ZNSt12_Vector_baseIiSaIiEED1Ev>
  405659:	48 89 d8             	mov    %rbx,%rax
  40565c:	48 89 c7             	mov    %rax,%rdi
  40565f:	e8 dc c1 ff ff       	callq  401840 <_Unwind_Resume@plt>
  405664:	48 83 c4 18          	add    $0x18,%rsp
  405668:	5b                   	pop    %rbx
  405669:	5d                   	pop    %rbp
  40566a:	c3                   	retq   
  40566b:	90                   	nop

000000000040566c <_ZNSt6vectorIN2cv3MatESaIS1_EEC1Ev>:
  40566c:	55                   	push   %rbp
  40566d:	48 89 e5             	mov    %rsp,%rbp
  405670:	48 83 ec 10          	sub    $0x10,%rsp
  405674:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  405678:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40567c:	48 89 c7             	mov    %rax,%rdi
  40567f:	e8 ac 02 00 00       	callq  405930 <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EEC1Ev>
  405684:	c9                   	leaveq 
  405685:	c3                   	retq   

0000000000405686 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev>:
  405686:	55                   	push   %rbp
  405687:	48 89 e5             	mov    %rsp,%rbp
  40568a:	53                   	push   %rbx
  40568b:	48 83 ec 18          	sub    $0x18,%rsp
  40568f:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  405693:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405697:	48 89 c7             	mov    %rax,%rdi
  40569a:	e8 2d 03 00 00       	callq  4059cc <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE19_M_get_Tp_allocatorEv>
  40569f:	48 89 c2             	mov    %rax,%rdx
  4056a2:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4056a6:	48 8b 48 08          	mov    0x8(%rax),%rcx
  4056aa:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4056ae:	48 8b 00             	mov    (%rax),%rax
  4056b1:	48 89 ce             	mov    %rcx,%rsi
  4056b4:	48 89 c7             	mov    %rax,%rdi
  4056b7:	e8 1e 03 00 00       	callq  4059da <_ZSt8_DestroyIPN2cv3MatES1_EvT_S3_RSaIT0_E>
  4056bc:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4056c0:	48 89 c7             	mov    %rax,%rdi
  4056c3:	e8 82 02 00 00       	callq  40594a <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EED1Ev>
  4056c8:	eb 1a                	jmp    4056e4 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x5e>
  4056ca:	48 89 c3             	mov    %rax,%rbx
  4056cd:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4056d1:	48 89 c7             	mov    %rax,%rdi
  4056d4:	e8 71 02 00 00       	callq  40594a <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EED1Ev>
  4056d9:	48 89 d8             	mov    %rbx,%rax
  4056dc:	48 89 c7             	mov    %rax,%rdi
  4056df:	e8 5c c1 ff ff       	callq  401840 <_Unwind_Resume@plt>
  4056e4:	48 83 c4 18          	add    $0x18,%rsp
  4056e8:	5b                   	pop    %rbx
  4056e9:	5d                   	pop    %rbp
  4056ea:	c3                   	retq   
  4056eb:	90                   	nop

00000000004056ec <_ZNSt6vectorIfSaIfEEC1Ev>:
  4056ec:	55                   	push   %rbp
  4056ed:	48 89 e5             	mov    %rsp,%rbp
  4056f0:	48 83 ec 10          	sub    $0x10,%rsp
  4056f4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4056f8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4056fc:	48 89 c7             	mov    %rax,%rdi
  4056ff:	e8 1a 03 00 00       	callq  405a1e <_ZNSt12_Vector_baseIfSaIfEEC1Ev>
  405704:	c9                   	leaveq 
  405705:	c3                   	retq   

0000000000405706 <_ZNSt6vectorIfSaIfEED1Ev>:
  405706:	55                   	push   %rbp
  405707:	48 89 e5             	mov    %rsp,%rbp
  40570a:	53                   	push   %rbx
  40570b:	48 83 ec 18          	sub    $0x18,%rsp
  40570f:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  405713:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405717:	48 89 c7             	mov    %rax,%rdi
  40571a:	e8 8b 03 00 00       	callq  405aaa <_ZNSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv>
  40571f:	48 89 c2             	mov    %rax,%rdx
  405722:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405726:	48 8b 48 08          	mov    0x8(%rax),%rcx
  40572a:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40572e:	48 8b 00             	mov    (%rax),%rax
  405731:	48 89 ce             	mov    %rcx,%rsi
  405734:	48 89 c7             	mov    %rax,%rdi
  405737:	e8 7c 03 00 00       	callq  405ab8 <_ZSt8_DestroyIPffEvT_S1_RSaIT0_E>
  40573c:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405740:	48 89 c7             	mov    %rax,%rdi
  405743:	e8 f0 02 00 00       	callq  405a38 <_ZNSt12_Vector_baseIfSaIfEED1Ev>
  405748:	eb 1a                	jmp    405764 <_ZNSt6vectorIfSaIfEED1Ev+0x5e>
  40574a:	48 89 c3             	mov    %rax,%rbx
  40574d:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405751:	48 89 c7             	mov    %rax,%rdi
  405754:	e8 df 02 00 00       	callq  405a38 <_ZNSt12_Vector_baseIfSaIfEED1Ev>
  405759:	48 89 d8             	mov    %rbx,%rax
  40575c:	48 89 c7             	mov    %rax,%rdi
  40575f:	e8 dc c0 ff ff       	callq  401840 <_Unwind_Resume@plt>
  405764:	48 83 c4 18          	add    $0x18,%rsp
  405768:	5b                   	pop    %rbx
  405769:	5d                   	pop    %rbp
  40576a:	c3                   	retq   
  40576b:	90                   	nop

000000000040576c <_ZNSt6vectorIfSaIfEE6assignIPhEEvT_S4_>:
  40576c:	55                   	push   %rbp
  40576d:	48 89 e5             	mov    %rsp,%rbp
  405770:	48 83 ec 30          	sub    $0x30,%rsp
  405774:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  405778:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  40577c:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
  405780:	48 8b 55 d8          	mov    -0x28(%rbp),%rdx
  405784:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  405788:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40578c:	44 88 04 24          	mov    %r8b,(%rsp)
  405790:	48 89 ce             	mov    %rcx,%rsi
  405793:	48 89 c7             	mov    %rax,%rdi
  405796:	e8 47 03 00 00       	callq  405ae2 <_ZNSt6vectorIfSaIfEE18_M_assign_dispatchIPhEEvT_S4_St12__false_type>
  40579b:	c9                   	leaveq 
  40579c:	c3                   	retq   
  40579d:	90                   	nop

000000000040579e <_ZNSt6vectorIfSaIfEEixEm>:
  40579e:	55                   	push   %rbp
  40579f:	48 89 e5             	mov    %rsp,%rbp
  4057a2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4057a6:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4057aa:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4057ae:	48 8b 00             	mov    (%rax),%rax
  4057b1:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4057b5:	48 c1 e2 02          	shl    $0x2,%rdx
  4057b9:	48 01 d0             	add    %rdx,%rax
  4057bc:	5d                   	pop    %rbp
  4057bd:	c3                   	retq   

00000000004057be <_ZNSt6vectorIN2cv3MatESaIS1_EE9push_backERKS1_>:
  4057be:	55                   	push   %rbp
  4057bf:	48 89 e5             	mov    %rsp,%rbp
  4057c2:	48 83 ec 10          	sub    $0x10,%rsp
  4057c6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4057ca:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4057ce:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4057d2:	48 8b 50 08          	mov    0x8(%rax),%rdx
  4057d6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4057da:	48 8b 40 10          	mov    0x10(%rax),%rax
  4057de:	48 39 c2             	cmp    %rax,%rdx
  4057e1:	74 31                	je     405814 <_ZNSt6vectorIN2cv3MatESaIS1_EE9push_backERKS1_+0x56>
  4057e3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4057e7:	48 8b 48 08          	mov    0x8(%rax),%rcx
  4057eb:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4057ef:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4057f3:	48 89 ce             	mov    %rcx,%rsi
  4057f6:	48 89 c7             	mov    %rax,%rdi
  4057f9:	e8 15 03 00 00       	callq  405b13 <_ZN9__gnu_cxx14__alloc_traitsISaIN2cv3MatEEE9constructIS2_EEvRS3_PS2_RKT_>
  4057fe:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405802:	48 8b 40 08          	mov    0x8(%rax),%rax
  405806:	48 8d 50 60          	lea    0x60(%rax),%rdx
  40580a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40580e:	48 89 50 08          	mov    %rdx,0x8(%rax)
  405812:	eb 22                	jmp    405836 <_ZNSt6vectorIN2cv3MatESaIS1_EE9push_backERKS1_+0x78>
  405814:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405818:	48 89 c7             	mov    %rax,%rdi
  40581b:	e8 f6 06 00 00       	callq  405f16 <_ZNSt6vectorIN2cv3MatESaIS1_EE3endEv>
  405820:	48 89 c1             	mov    %rax,%rcx
  405823:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  405827:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40582b:	48 89 ce             	mov    %rcx,%rsi
  40582e:	48 89 c7             	mov    %rax,%rdi
  405831:	e8 0a 03 00 00       	callq  405b40 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_>
  405836:	c9                   	leaveq 
  405837:	c3                   	retq   

0000000000405838 <_ZNSt12_Vector_baseIiSaIiEE12_Vector_implD1Ev>:
  405838:	55                   	push   %rbp
  405839:	48 89 e5             	mov    %rsp,%rbp
  40583c:	48 83 ec 10          	sub    $0x10,%rsp
  405840:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  405844:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405848:	48 89 c7             	mov    %rax,%rdi
  40584b:	e8 2e 07 00 00       	callq  405f7e <_ZNSaIiED1Ev>
  405850:	c9                   	leaveq 
  405851:	c3                   	retq   

0000000000405852 <_ZNSt12_Vector_baseIiSaIiEEC1Ev>:
  405852:	55                   	push   %rbp
  405853:	48 89 e5             	mov    %rsp,%rbp
  405856:	48 83 ec 10          	sub    $0x10,%rsp
  40585a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40585e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405862:	48 89 c7             	mov    %rax,%rdi
  405865:	e8 d6 06 00 00       	callq  405f40 <_ZNSt12_Vector_baseIiSaIiEE12_Vector_implC1Ev>
  40586a:	c9                   	leaveq 
  40586b:	c3                   	retq   

000000000040586c <_ZNSt12_Vector_baseIiSaIiEED1Ev>:
  40586c:	55                   	push   %rbp
  40586d:	48 89 e5             	mov    %rsp,%rbp
  405870:	53                   	push   %rbx
  405871:	48 83 ec 18          	sub    $0x18,%rsp
  405875:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  405879:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40587d:	48 8b 40 10          	mov    0x10(%rax),%rax
  405881:	48 89 c2             	mov    %rax,%rdx
  405884:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405888:	48 8b 00             	mov    (%rax),%rax
  40588b:	48 29 c2             	sub    %rax,%rdx
  40588e:	48 89 d0             	mov    %rdx,%rax
  405891:	48 c1 f8 02          	sar    $0x2,%rax
  405895:	48 89 c2             	mov    %rax,%rdx
  405898:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40589c:	48 8b 08             	mov    (%rax),%rcx
  40589f:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4058a3:	48 89 ce             	mov    %rcx,%rsi
  4058a6:	48 89 c7             	mov    %rax,%rdi
  4058a9:	e8 ea 06 00 00       	callq  405f98 <_ZNSt12_Vector_baseIiSaIiEE13_M_deallocateEPim>
  4058ae:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4058b2:	48 89 c7             	mov    %rax,%rdi
  4058b5:	e8 7e ff ff ff       	callq  405838 <_ZNSt12_Vector_baseIiSaIiEE12_Vector_implD1Ev>
  4058ba:	eb 1a                	jmp    4058d6 <_ZNSt12_Vector_baseIiSaIiEED1Ev+0x6a>
  4058bc:	48 89 c3             	mov    %rax,%rbx
  4058bf:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4058c3:	48 89 c7             	mov    %rax,%rdi
  4058c6:	e8 6d ff ff ff       	callq  405838 <_ZNSt12_Vector_baseIiSaIiEE12_Vector_implD1Ev>
  4058cb:	48 89 d8             	mov    %rbx,%rax
  4058ce:	48 89 c7             	mov    %rax,%rdi
  4058d1:	e8 6a bf ff ff       	callq  401840 <_Unwind_Resume@plt>
  4058d6:	48 83 c4 18          	add    $0x18,%rsp
  4058da:	5b                   	pop    %rbx
  4058db:	5d                   	pop    %rbp
  4058dc:	c3                   	retq   
  4058dd:	90                   	nop

00000000004058de <_ZNSt12_Vector_baseIiSaIiEE19_M_get_Tp_allocatorEv>:
  4058de:	55                   	push   %rbp
  4058df:	48 89 e5             	mov    %rsp,%rbp
  4058e2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4058e6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4058ea:	5d                   	pop    %rbp
  4058eb:	c3                   	retq   

00000000004058ec <_ZSt8_DestroyIPiiEvT_S1_RSaIT0_E>:
  4058ec:	55                   	push   %rbp
  4058ed:	48 89 e5             	mov    %rsp,%rbp
  4058f0:	48 83 ec 20          	sub    $0x20,%rsp
  4058f4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4058f8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4058fc:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  405900:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  405904:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405908:	48 89 d6             	mov    %rdx,%rsi
  40590b:	48 89 c7             	mov    %rax,%rdi
  40590e:	e8 b9 06 00 00       	callq  405fcc <_ZSt8_DestroyIPiEvT_S1_>
  405913:	c9                   	leaveq 
  405914:	c3                   	retq   
  405915:	90                   	nop

0000000000405916 <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE12_Vector_implD1Ev>:
  405916:	55                   	push   %rbp
  405917:	48 89 e5             	mov    %rsp,%rbp
  40591a:	48 83 ec 10          	sub    $0x10,%rsp
  40591e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  405922:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405926:	48 89 c7             	mov    %rax,%rdi
  405929:	e8 02 07 00 00       	callq  406030 <_ZNSaIN2cv3MatEED1Ev>
  40592e:	c9                   	leaveq 
  40592f:	c3                   	retq   

0000000000405930 <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EEC1Ev>:
  405930:	55                   	push   %rbp
  405931:	48 89 e5             	mov    %rsp,%rbp
  405934:	48 83 ec 10          	sub    $0x10,%rsp
  405938:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40593c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405940:	48 89 c7             	mov    %rax,%rdi
  405943:	e8 aa 06 00 00       	callq  405ff2 <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE12_Vector_implC1Ev>
  405948:	c9                   	leaveq 
  405949:	c3                   	retq   

000000000040594a <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EED1Ev>:
  40594a:	55                   	push   %rbp
  40594b:	48 89 e5             	mov    %rsp,%rbp
  40594e:	53                   	push   %rbx
  40594f:	48 83 ec 18          	sub    $0x18,%rsp
  405953:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  405957:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40595b:	48 8b 40 10          	mov    0x10(%rax),%rax
  40595f:	48 89 c2             	mov    %rax,%rdx
  405962:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405966:	48 8b 00             	mov    (%rax),%rax
  405969:	48 29 c2             	sub    %rax,%rdx
  40596c:	48 89 d0             	mov    %rdx,%rax
  40596f:	48 c1 f8 05          	sar    $0x5,%rax
  405973:	48 89 c2             	mov    %rax,%rdx
  405976:	48 b8 ab aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaab,%rax
  40597d:	aa aa aa 
  405980:	48 0f af c2          	imul   %rdx,%rax
  405984:	48 89 c2             	mov    %rax,%rdx
  405987:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40598b:	48 8b 08             	mov    (%rax),%rcx
  40598e:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405992:	48 89 ce             	mov    %rcx,%rsi
  405995:	48 89 c7             	mov    %rax,%rdi
  405998:	e8 ad 06 00 00       	callq  40604a <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE13_M_deallocateEPS1_m>
  40599d:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4059a1:	48 89 c7             	mov    %rax,%rdi
  4059a4:	e8 6d ff ff ff       	callq  405916 <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE12_Vector_implD1Ev>
  4059a9:	eb 1a                	jmp    4059c5 <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EED1Ev+0x7b>
  4059ab:	48 89 c3             	mov    %rax,%rbx
  4059ae:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4059b2:	48 89 c7             	mov    %rax,%rdi
  4059b5:	e8 5c ff ff ff       	callq  405916 <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE12_Vector_implD1Ev>
  4059ba:	48 89 d8             	mov    %rbx,%rax
  4059bd:	48 89 c7             	mov    %rax,%rdi
  4059c0:	e8 7b be ff ff       	callq  401840 <_Unwind_Resume@plt>
  4059c5:	48 83 c4 18          	add    $0x18,%rsp
  4059c9:	5b                   	pop    %rbx
  4059ca:	5d                   	pop    %rbp
  4059cb:	c3                   	retq   

00000000004059cc <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE19_M_get_Tp_allocatorEv>:
  4059cc:	55                   	push   %rbp
  4059cd:	48 89 e5             	mov    %rsp,%rbp
  4059d0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4059d4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4059d8:	5d                   	pop    %rbp
  4059d9:	c3                   	retq   

00000000004059da <_ZSt8_DestroyIPN2cv3MatES1_EvT_S3_RSaIT0_E>:
  4059da:	55                   	push   %rbp
  4059db:	48 89 e5             	mov    %rsp,%rbp
  4059de:	48 83 ec 20          	sub    $0x20,%rsp
  4059e2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4059e6:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4059ea:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  4059ee:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4059f2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4059f6:	48 89 d6             	mov    %rdx,%rsi
  4059f9:	48 89 c7             	mov    %rax,%rdi
  4059fc:	e8 7d 06 00 00       	callq  40607e <_ZSt8_DestroyIPN2cv3MatEEvT_S3_>
  405a01:	c9                   	leaveq 
  405a02:	c3                   	retq   
  405a03:	90                   	nop

0000000000405a04 <_ZNSt12_Vector_baseIfSaIfEE12_Vector_implD1Ev>:
  405a04:	55                   	push   %rbp
  405a05:	48 89 e5             	mov    %rsp,%rbp
  405a08:	48 83 ec 10          	sub    $0x10,%rsp
  405a0c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  405a10:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405a14:	48 89 c7             	mov    %rax,%rdi
  405a17:	e8 c6 06 00 00       	callq  4060e2 <_ZNSaIfED1Ev>
  405a1c:	c9                   	leaveq 
  405a1d:	c3                   	retq   

0000000000405a1e <_ZNSt12_Vector_baseIfSaIfEEC1Ev>:
  405a1e:	55                   	push   %rbp
  405a1f:	48 89 e5             	mov    %rsp,%rbp
  405a22:	48 83 ec 10          	sub    $0x10,%rsp
  405a26:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  405a2a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405a2e:	48 89 c7             	mov    %rax,%rdi
  405a31:	e8 6e 06 00 00       	callq  4060a4 <_ZNSt12_Vector_baseIfSaIfEE12_Vector_implC1Ev>
  405a36:	c9                   	leaveq 
  405a37:	c3                   	retq   

0000000000405a38 <_ZNSt12_Vector_baseIfSaIfEED1Ev>:
  405a38:	55                   	push   %rbp
  405a39:	48 89 e5             	mov    %rsp,%rbp
  405a3c:	53                   	push   %rbx
  405a3d:	48 83 ec 18          	sub    $0x18,%rsp
  405a41:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  405a45:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405a49:	48 8b 40 10          	mov    0x10(%rax),%rax
  405a4d:	48 89 c2             	mov    %rax,%rdx
  405a50:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405a54:	48 8b 00             	mov    (%rax),%rax
  405a57:	48 29 c2             	sub    %rax,%rdx
  405a5a:	48 89 d0             	mov    %rdx,%rax
  405a5d:	48 c1 f8 02          	sar    $0x2,%rax
  405a61:	48 89 c2             	mov    %rax,%rdx
  405a64:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405a68:	48 8b 08             	mov    (%rax),%rcx
  405a6b:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405a6f:	48 89 ce             	mov    %rcx,%rsi
  405a72:	48 89 c7             	mov    %rax,%rdi
  405a75:	e8 82 06 00 00       	callq  4060fc <_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm>
  405a7a:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405a7e:	48 89 c7             	mov    %rax,%rdi
  405a81:	e8 7e ff ff ff       	callq  405a04 <_ZNSt12_Vector_baseIfSaIfEE12_Vector_implD1Ev>
  405a86:	eb 1a                	jmp    405aa2 <_ZNSt12_Vector_baseIfSaIfEED1Ev+0x6a>
  405a88:	48 89 c3             	mov    %rax,%rbx
  405a8b:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405a8f:	48 89 c7             	mov    %rax,%rdi
  405a92:	e8 6d ff ff ff       	callq  405a04 <_ZNSt12_Vector_baseIfSaIfEE12_Vector_implD1Ev>
  405a97:	48 89 d8             	mov    %rbx,%rax
  405a9a:	48 89 c7             	mov    %rax,%rdi
  405a9d:	e8 9e bd ff ff       	callq  401840 <_Unwind_Resume@plt>
  405aa2:	48 83 c4 18          	add    $0x18,%rsp
  405aa6:	5b                   	pop    %rbx
  405aa7:	5d                   	pop    %rbp
  405aa8:	c3                   	retq   
  405aa9:	90                   	nop

0000000000405aaa <_ZNSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv>:
  405aaa:	55                   	push   %rbp
  405aab:	48 89 e5             	mov    %rsp,%rbp
  405aae:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  405ab2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405ab6:	5d                   	pop    %rbp
  405ab7:	c3                   	retq   

0000000000405ab8 <_ZSt8_DestroyIPffEvT_S1_RSaIT0_E>:
  405ab8:	55                   	push   %rbp
  405ab9:	48 89 e5             	mov    %rsp,%rbp
  405abc:	48 83 ec 20          	sub    $0x20,%rsp
  405ac0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  405ac4:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  405ac8:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  405acc:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  405ad0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405ad4:	48 89 d6             	mov    %rdx,%rsi
  405ad7:	48 89 c7             	mov    %rax,%rdi
  405ada:	e8 51 06 00 00       	callq  406130 <_ZSt8_DestroyIPfEvT_S1_>
  405adf:	c9                   	leaveq 
  405ae0:	c3                   	retq   
  405ae1:	90                   	nop

0000000000405ae2 <_ZNSt6vectorIfSaIfEE18_M_assign_dispatchIPhEEvT_S4_St12__false_type>:
  405ae2:	55                   	push   %rbp
  405ae3:	48 89 e5             	mov    %rsp,%rbp
  405ae6:	48 83 ec 30          	sub    $0x30,%rsp
  405aea:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  405aee:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  405af2:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
  405af6:	48 8b 55 d8          	mov    -0x28(%rbp),%rdx
  405afa:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  405afe:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405b02:	44 88 04 24          	mov    %r8b,(%rsp)
  405b06:	48 89 ce             	mov    %rcx,%rsi
  405b09:	48 89 c7             	mov    %rax,%rdi
  405b0c:	e8 45 06 00 00       	callq  406156 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag>
  405b11:	c9                   	leaveq 
  405b12:	c3                   	retq   

0000000000405b13 <_ZN9__gnu_cxx14__alloc_traitsISaIN2cv3MatEEE9constructIS2_EEvRS3_PS2_RKT_>:
  405b13:	55                   	push   %rbp
  405b14:	48 89 e5             	mov    %rsp,%rbp
  405b17:	48 83 ec 20          	sub    $0x20,%rsp
  405b1b:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  405b1f:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  405b23:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  405b27:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  405b2b:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  405b2f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405b33:	48 89 ce             	mov    %rcx,%rsi
  405b36:	48 89 c7             	mov    %rax,%rdi
  405b39:	e8 c8 07 00 00       	callq  406306 <_ZN9__gnu_cxx13new_allocatorIN2cv3MatEE9constructEPS2_RKS2_>
  405b3e:	c9                   	leaveq 
  405b3f:	c3                   	retq   

0000000000405b40 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_>:
  405b40:	55                   	push   %rbp
  405b41:	48 89 e5             	mov    %rsp,%rbp
  405b44:	41 54                	push   %r12
  405b46:	53                   	push   %rbx
  405b47:	48 81 ec b0 00 00 00 	sub    $0xb0,%rsp
  405b4e:	48 89 bd 58 ff ff ff 	mov    %rdi,-0xa8(%rbp)
  405b55:	48 89 b5 50 ff ff ff 	mov    %rsi,-0xb0(%rbp)
  405b5c:	48 89 95 48 ff ff ff 	mov    %rdx,-0xb8(%rbp)
  405b63:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405b6a:	48 8b 50 08          	mov    0x8(%rax),%rdx
  405b6e:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405b75:	48 8b 40 10          	mov    0x10(%rax),%rax
  405b79:	48 39 c2             	cmp    %rax,%rdx
  405b7c:	0f 84 d5 00 00 00    	je     405c57 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x117>
  405b82:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405b89:	48 8b 40 08          	mov    0x8(%rax),%rax
  405b8d:	48 8d 50 a0          	lea    -0x60(%rax),%rdx
  405b91:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405b98:	48 8b 48 08          	mov    0x8(%rax),%rcx
  405b9c:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405ba3:	48 89 ce             	mov    %rcx,%rsi
  405ba6:	48 89 c7             	mov    %rax,%rdi
  405ba9:	e8 65 ff ff ff       	callq  405b13 <_ZN9__gnu_cxx14__alloc_traitsISaIN2cv3MatEEE9constructIS2_EEvRS3_PS2_RKT_>
  405bae:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405bb5:	48 8b 40 08          	mov    0x8(%rax),%rax
  405bb9:	48 8d 50 60          	lea    0x60(%rax),%rdx
  405bbd:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405bc4:	48 89 50 08          	mov    %rdx,0x8(%rax)
  405bc8:	48 8b 95 48 ff ff ff 	mov    -0xb8(%rbp),%rdx
  405bcf:	48 8d 85 60 ff ff ff 	lea    -0xa0(%rbp),%rax
  405bd6:	48 89 d6             	mov    %rdx,%rsi
  405bd9:	48 89 c7             	mov    %rax,%rdi
  405bdc:	e8 67 f3 ff ff       	callq  404f48 <_ZN2cv3MatC1ERKS0_>
  405be1:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405be8:	48 8b 40 08          	mov    0x8(%rax),%rax
  405bec:	4c 8d 60 a0          	lea    -0x60(%rax),%r12
  405bf0:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405bf7:	48 8b 40 08          	mov    0x8(%rax),%rax
  405bfb:	48 8d 98 40 ff ff ff 	lea    -0xc0(%rax),%rbx
  405c02:	48 8d 85 50 ff ff ff 	lea    -0xb0(%rbp),%rax
  405c09:	48 89 c7             	mov    %rax,%rdi
  405c0c:	e8 5f 07 00 00       	callq  406370 <_ZNK9__gnu_cxx17__normal_iteratorIPN2cv3MatESt6vectorIS2_SaIS2_EEE4baseEv>
  405c11:	48 8b 00             	mov    (%rax),%rax
  405c14:	4c 89 e2             	mov    %r12,%rdx
  405c17:	48 89 de             	mov    %rbx,%rsi
  405c1a:	48 89 c7             	mov    %rax,%rdi
  405c1d:	e8 5c 07 00 00       	callq  40637e <_ZSt13copy_backwardIPN2cv3MatES2_ET0_T_S4_S3_>
  405c22:	48 8d 85 50 ff ff ff 	lea    -0xb0(%rbp),%rax
  405c29:	48 89 c7             	mov    %rax,%rdi
  405c2c:	e8 93 07 00 00       	callq  4063c4 <_ZNK9__gnu_cxx17__normal_iteratorIPN2cv3MatESt6vectorIS2_SaIS2_EEEdeEv>
  405c31:	48 8d 95 60 ff ff ff 	lea    -0xa0(%rbp),%rdx
  405c38:	48 89 d6             	mov    %rdx,%rsi
  405c3b:	48 89 c7             	mov    %rax,%rdi
  405c3e:	e8 b9 f6 ff ff       	callq  4052fc <_ZN2cv3MataSERKS0_>
  405c43:	48 8d 85 60 ff ff ff 	lea    -0xa0(%rbp),%rax
  405c4a:	48 89 c7             	mov    %rax,%rdi
  405c4d:	e8 6a f6 ff ff       	callq  4052bc <_ZN2cv3MatD1Ev>
  405c52:	e9 b2 02 00 00       	jmpq   405f09 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x3c9>
  405c57:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405c5e:	ba 48 71 40 00       	mov    $0x407148,%edx
  405c63:	be 01 00 00 00       	mov    $0x1,%esi
  405c68:	48 89 c7             	mov    %rax,%rdi
  405c6b:	e8 66 07 00 00       	callq  4063d6 <_ZNKSt6vectorIN2cv3MatESaIS1_EE12_M_check_lenEmPKc>
  405c70:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  405c74:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405c7b:	48 89 c7             	mov    %rax,%rdi
  405c7e:	e8 1d 08 00 00       	callq  4064a0 <_ZNSt6vectorIN2cv3MatESaIS1_EE5beginEv>
  405c83:	48 89 45 c0          	mov    %rax,-0x40(%rbp)
  405c87:	48 8d 55 c0          	lea    -0x40(%rbp),%rdx
  405c8b:	48 8d 85 50 ff ff ff 	lea    -0xb0(%rbp),%rax
  405c92:	48 89 d6             	mov    %rdx,%rsi
  405c95:	48 89 c7             	mov    %rax,%rdi
  405c98:	e8 28 08 00 00       	callq  4064c5 <_ZN9__gnu_cxxmiIPN2cv3MatESt6vectorIS2_SaIS2_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKSA_SD_>
  405c9d:	48 89 45 d8          	mov    %rax,-0x28(%rbp)
  405ca1:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405ca8:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
  405cac:	48 89 d6             	mov    %rdx,%rsi
  405caf:	48 89 c7             	mov    %rax,%rdi
  405cb2:	e8 63 08 00 00       	callq  40651a <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE11_M_allocateEm>
  405cb7:	48 89 45 d0          	mov    %rax,-0x30(%rbp)
  405cbb:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  405cbf:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  405cc3:	48 8b 55 d8          	mov    -0x28(%rbp),%rdx
  405cc7:	48 89 d0             	mov    %rdx,%rax
  405cca:	48 01 c0             	add    %rax,%rax
  405ccd:	48 01 d0             	add    %rdx,%rax
  405cd0:	48 c1 e0 05          	shl    $0x5,%rax
  405cd4:	48 89 c2             	mov    %rax,%rdx
  405cd7:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  405cdb:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  405cdf:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405ce6:	48 8b 95 48 ff ff ff 	mov    -0xb8(%rbp),%rdx
  405ced:	48 89 ce             	mov    %rcx,%rsi
  405cf0:	48 89 c7             	mov    %rax,%rdi
  405cf3:	e8 1b fe ff ff       	callq  405b13 <_ZN9__gnu_cxx14__alloc_traitsISaIN2cv3MatEEE9constructIS2_EEvRS3_PS2_RKT_>
  405cf8:	48 c7 45 e8 00 00 00 	movq   $0x0,-0x18(%rbp)
  405cff:	00 
  405d00:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405d07:	48 89 c7             	mov    %rax,%rdi
  405d0a:	e8 bd fc ff ff       	callq  4059cc <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE19_M_get_Tp_allocatorEv>
  405d0f:	48 89 c3             	mov    %rax,%rbx
  405d12:	48 8d 85 50 ff ff ff 	lea    -0xb0(%rbp),%rax
  405d19:	48 89 c7             	mov    %rax,%rdi
  405d1c:	e8 4f 06 00 00       	callq  406370 <_ZNK9__gnu_cxx17__normal_iteratorIPN2cv3MatESt6vectorIS2_SaIS2_EEE4baseEv>
  405d21:	48 8b 30             	mov    (%rax),%rsi
  405d24:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405d2b:	48 8b 00             	mov    (%rax),%rax
  405d2e:	48 8b 55 d0          	mov    -0x30(%rbp),%rdx
  405d32:	48 89 d9             	mov    %rbx,%rcx
  405d35:	48 89 c7             	mov    %rax,%rdi
  405d38:	e8 15 08 00 00       	callq  406552 <_ZSt34__uninitialized_move_if_noexcept_aIPN2cv3MatES2_SaIS1_EET0_T_S5_S4_RT1_>
  405d3d:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  405d41:	48 83 45 e8 60       	addq   $0x60,-0x18(%rbp)
  405d46:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405d4d:	48 89 c7             	mov    %rax,%rdi
  405d50:	e8 77 fc ff ff       	callq  4059cc <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE19_M_get_Tp_allocatorEv>
  405d55:	49 89 c4             	mov    %rax,%r12
  405d58:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405d5f:	48 8b 58 08          	mov    0x8(%rax),%rbx
  405d63:	48 8d 85 50 ff ff ff 	lea    -0xb0(%rbp),%rax
  405d6a:	48 89 c7             	mov    %rax,%rdi
  405d6d:	e8 fe 05 00 00       	callq  406370 <_ZNK9__gnu_cxx17__normal_iteratorIPN2cv3MatESt6vectorIS2_SaIS2_EEE4baseEv>
  405d72:	48 8b 00             	mov    (%rax),%rax
  405d75:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  405d79:	4c 89 e1             	mov    %r12,%rcx
  405d7c:	48 89 de             	mov    %rbx,%rsi
  405d7f:	48 89 c7             	mov    %rax,%rdi
  405d82:	e8 cb 07 00 00       	callq  406552 <_ZSt34__uninitialized_move_if_noexcept_aIPN2cv3MatES2_SaIS1_EET0_T_S5_S4_RT1_>
  405d87:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  405d8b:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405d92:	48 89 c7             	mov    %rax,%rdi
  405d95:	e8 32 fc ff ff       	callq  4059cc <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE19_M_get_Tp_allocatorEv>
  405d9a:	48 89 c2             	mov    %rax,%rdx
  405d9d:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405da4:	48 8b 48 08          	mov    0x8(%rax),%rcx
  405da8:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405daf:	48 8b 00             	mov    (%rax),%rax
  405db2:	48 89 ce             	mov    %rcx,%rsi
  405db5:	48 89 c7             	mov    %rax,%rdi
  405db8:	e8 1d fc ff ff       	callq  4059da <_ZSt8_DestroyIPN2cv3MatES1_EvT_S3_RSaIT0_E>
  405dbd:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405dc4:	48 8b 40 10          	mov    0x10(%rax),%rax
  405dc8:	48 89 c2             	mov    %rax,%rdx
  405dcb:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405dd2:	48 8b 00             	mov    (%rax),%rax
  405dd5:	48 29 c2             	sub    %rax,%rdx
  405dd8:	48 89 d0             	mov    %rdx,%rax
  405ddb:	48 c1 f8 05          	sar    $0x5,%rax
  405ddf:	48 89 c2             	mov    %rax,%rdx
  405de2:	48 b8 ab aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaab,%rax
  405de9:	aa aa aa 
  405dec:	48 0f af c2          	imul   %rdx,%rax
  405df0:	48 89 c2             	mov    %rax,%rdx
  405df3:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405dfa:	48 8b 08             	mov    (%rax),%rcx
  405dfd:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405e04:	48 89 ce             	mov    %rcx,%rsi
  405e07:	48 89 c7             	mov    %rax,%rdi
  405e0a:	e8 3b 02 00 00       	callq  40604a <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE13_M_deallocateEPS1_m>
  405e0f:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405e16:	48 8b 55 d0          	mov    -0x30(%rbp),%rdx
  405e1a:	48 89 10             	mov    %rdx,(%rax)
  405e1d:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405e24:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  405e28:	48 89 50 08          	mov    %rdx,0x8(%rax)
  405e2c:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
  405e30:	48 89 d0             	mov    %rdx,%rax
  405e33:	48 01 c0             	add    %rax,%rax
  405e36:	48 01 d0             	add    %rdx,%rax
  405e39:	48 c1 e0 05          	shl    $0x5,%rax
  405e3d:	48 89 c2             	mov    %rax,%rdx
  405e40:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  405e44:	48 01 c2             	add    %rax,%rdx
  405e47:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405e4e:	48 89 50 10          	mov    %rdx,0x10(%rax)
  405e52:	e9 b2 00 00 00       	jmpq   405f09 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x3c9>
  405e57:	48 89 c3             	mov    %rax,%rbx
  405e5a:	48 8d 85 60 ff ff ff 	lea    -0xa0(%rbp),%rax
  405e61:	48 89 c7             	mov    %rax,%rdi
  405e64:	e8 53 f4 ff ff       	callq  4052bc <_ZN2cv3MatD1Ev>
  405e69:	48 89 d8             	mov    %rbx,%rax
  405e6c:	48 89 c7             	mov    %rax,%rdi
  405e6f:	e8 cc b9 ff ff       	callq  401840 <_Unwind_Resume@plt>
  405e74:	48 89 c7             	mov    %rax,%rdi
  405e77:	e8 84 b8 ff ff       	callq  401700 <__cxa_begin_catch@plt>
  405e7c:	48 83 7d e8 00       	cmpq   $0x0,-0x18(%rbp)
  405e81:	75 2f                	jne    405eb2 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x372>
  405e83:	48 8b 55 d8          	mov    -0x28(%rbp),%rdx
  405e87:	48 89 d0             	mov    %rdx,%rax
  405e8a:	48 01 c0             	add    %rax,%rax
  405e8d:	48 01 d0             	add    %rdx,%rax
  405e90:	48 c1 e0 05          	shl    $0x5,%rax
  405e94:	48 89 c2             	mov    %rax,%rdx
  405e97:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  405e9b:	48 01 c2             	add    %rax,%rdx
  405e9e:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405ea5:	48 89 d6             	mov    %rdx,%rsi
  405ea8:	48 89 c7             	mov    %rax,%rdi
  405eab:	e8 d4 06 00 00       	callq  406584 <_ZN9__gnu_cxx14__alloc_traitsISaIN2cv3MatEEE7destroyERS3_PS2_>
  405eb0:	eb 25                	jmp    405ed7 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x397>
  405eb2:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405eb9:	48 89 c7             	mov    %rax,%rdi
  405ebc:	e8 0b fb ff ff       	callq  4059cc <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE19_M_get_Tp_allocatorEv>
  405ec1:	48 89 c2             	mov    %rax,%rdx
  405ec4:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  405ec8:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  405ecc:	48 89 ce             	mov    %rcx,%rsi
  405ecf:	48 89 c7             	mov    %rax,%rdi
  405ed2:	e8 03 fb ff ff       	callq  4059da <_ZSt8_DestroyIPN2cv3MatES1_EvT_S3_RSaIT0_E>
  405ed7:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
  405ede:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
  405ee2:	48 8b 4d d0          	mov    -0x30(%rbp),%rcx
  405ee6:	48 89 ce             	mov    %rcx,%rsi
  405ee9:	48 89 c7             	mov    %rax,%rdi
  405eec:	e8 59 01 00 00       	callq  40604a <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE13_M_deallocateEPS1_m>
  405ef1:	e8 ba b9 ff ff       	callq  4018b0 <__cxa_rethrow@plt>
  405ef6:	48 89 c3             	mov    %rax,%rbx
  405ef9:	e8 92 b9 ff ff       	callq  401890 <__cxa_end_catch@plt>
  405efe:	48 89 d8             	mov    %rbx,%rax
  405f01:	48 89 c7             	mov    %rax,%rdi
  405f04:	e8 37 b9 ff ff       	callq  401840 <_Unwind_Resume@plt>
  405f09:	48 81 c4 b0 00 00 00 	add    $0xb0,%rsp
  405f10:	5b                   	pop    %rbx
  405f11:	41 5c                	pop    %r12
  405f13:	5d                   	pop    %rbp
  405f14:	c3                   	retq   
  405f15:	90                   	nop

0000000000405f16 <_ZNSt6vectorIN2cv3MatESaIS1_EE3endEv>:
  405f16:	55                   	push   %rbp
  405f17:	48 89 e5             	mov    %rsp,%rbp
  405f1a:	48 83 ec 20          	sub    $0x20,%rsp
  405f1e:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  405f22:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  405f26:	48 8d 50 08          	lea    0x8(%rax),%rdx
  405f2a:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  405f2e:	48 89 d6             	mov    %rdx,%rsi
  405f31:	48 89 c7             	mov    %rax,%rdi
  405f34:	e8 71 06 00 00       	callq  4065aa <_ZN9__gnu_cxx17__normal_iteratorIPN2cv3MatESt6vectorIS2_SaIS2_EEEC1ERKS3_>
  405f39:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  405f3d:	c9                   	leaveq 
  405f3e:	c3                   	retq   
  405f3f:	90                   	nop

0000000000405f40 <_ZNSt12_Vector_baseIiSaIiEE12_Vector_implC1Ev>:
  405f40:	55                   	push   %rbp
  405f41:	48 89 e5             	mov    %rsp,%rbp
  405f44:	48 83 ec 10          	sub    $0x10,%rsp
  405f48:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  405f4c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405f50:	48 89 c7             	mov    %rax,%rdi
  405f53:	e8 6e 06 00 00       	callq  4065c6 <_ZNSaIiEC1Ev>
  405f58:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405f5c:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  405f63:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405f67:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  405f6e:	00 
  405f6f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405f73:	48 c7 40 10 00 00 00 	movq   $0x0,0x10(%rax)
  405f7a:	00 
  405f7b:	c9                   	leaveq 
  405f7c:	c3                   	retq   
  405f7d:	90                   	nop

0000000000405f7e <_ZNSaIiED1Ev>:
  405f7e:	55                   	push   %rbp
  405f7f:	48 89 e5             	mov    %rsp,%rbp
  405f82:	48 83 ec 10          	sub    $0x10,%rsp
  405f86:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  405f8a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405f8e:	48 89 c7             	mov    %rax,%rdi
  405f91:	e8 4a 06 00 00       	callq  4065e0 <_ZN9__gnu_cxx13new_allocatorIiED1Ev>
  405f96:	c9                   	leaveq 
  405f97:	c3                   	retq   

0000000000405f98 <_ZNSt12_Vector_baseIiSaIiEE13_M_deallocateEPim>:
  405f98:	55                   	push   %rbp
  405f99:	48 89 e5             	mov    %rsp,%rbp
  405f9c:	48 83 ec 20          	sub    $0x20,%rsp
  405fa0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  405fa4:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  405fa8:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  405fac:	48 83 7d f0 00       	cmpq   $0x0,-0x10(%rbp)
  405fb1:	74 17                	je     405fca <_ZNSt12_Vector_baseIiSaIiEE13_M_deallocateEPim+0x32>
  405fb3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405fb7:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  405fbb:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  405fbf:	48 89 ce             	mov    %rcx,%rsi
  405fc2:	48 89 c7             	mov    %rax,%rdi
  405fc5:	e8 20 06 00 00       	callq  4065ea <_ZN9__gnu_cxx13new_allocatorIiE10deallocateEPim>
  405fca:	c9                   	leaveq 
  405fcb:	c3                   	retq   

0000000000405fcc <_ZSt8_DestroyIPiEvT_S1_>:
  405fcc:	55                   	push   %rbp
  405fcd:	48 89 e5             	mov    %rsp,%rbp
  405fd0:	48 83 ec 10          	sub    $0x10,%rsp
  405fd4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  405fd8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  405fdc:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  405fe0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  405fe4:	48 89 d6             	mov    %rdx,%rsi
  405fe7:	48 89 c7             	mov    %rax,%rdi
  405fea:	e8 1d 06 00 00       	callq  40660c <_ZNSt12_Destroy_auxILb1EE9__destroyIPiEEvT_S3_>
  405fef:	c9                   	leaveq 
  405ff0:	c3                   	retq   
  405ff1:	90                   	nop

0000000000405ff2 <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE12_Vector_implC1Ev>:
  405ff2:	55                   	push   %rbp
  405ff3:	48 89 e5             	mov    %rsp,%rbp
  405ff6:	48 83 ec 10          	sub    $0x10,%rsp
  405ffa:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  405ffe:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406002:	48 89 c7             	mov    %rax,%rdi
  406005:	e8 10 06 00 00       	callq  40661a <_ZNSaIN2cv3MatEEC1Ev>
  40600a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40600e:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  406015:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406019:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  406020:	00 
  406021:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406025:	48 c7 40 10 00 00 00 	movq   $0x0,0x10(%rax)
  40602c:	00 
  40602d:	c9                   	leaveq 
  40602e:	c3                   	retq   
  40602f:	90                   	nop

0000000000406030 <_ZNSaIN2cv3MatEED1Ev>:
  406030:	55                   	push   %rbp
  406031:	48 89 e5             	mov    %rsp,%rbp
  406034:	48 83 ec 10          	sub    $0x10,%rsp
  406038:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40603c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406040:	48 89 c7             	mov    %rax,%rdi
  406043:	e8 ec 05 00 00       	callq  406634 <_ZN9__gnu_cxx13new_allocatorIN2cv3MatEED1Ev>
  406048:	c9                   	leaveq 
  406049:	c3                   	retq   

000000000040604a <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE13_M_deallocateEPS1_m>:
  40604a:	55                   	push   %rbp
  40604b:	48 89 e5             	mov    %rsp,%rbp
  40604e:	48 83 ec 20          	sub    $0x20,%rsp
  406052:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406056:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40605a:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  40605e:	48 83 7d f0 00       	cmpq   $0x0,-0x10(%rbp)
  406063:	74 17                	je     40607c <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE13_M_deallocateEPS1_m+0x32>
  406065:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406069:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  40606d:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  406071:	48 89 ce             	mov    %rcx,%rsi
  406074:	48 89 c7             	mov    %rax,%rdi
  406077:	e8 c2 05 00 00       	callq  40663e <_ZN9__gnu_cxx13new_allocatorIN2cv3MatEE10deallocateEPS2_m>
  40607c:	c9                   	leaveq 
  40607d:	c3                   	retq   

000000000040607e <_ZSt8_DestroyIPN2cv3MatEEvT_S3_>:
  40607e:	55                   	push   %rbp
  40607f:	48 89 e5             	mov    %rsp,%rbp
  406082:	48 83 ec 10          	sub    $0x10,%rsp
  406086:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40608a:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40608e:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  406092:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406096:	48 89 d6             	mov    %rdx,%rsi
  406099:	48 89 c7             	mov    %rax,%rdi
  40609c:	e8 bf 05 00 00       	callq  406660 <_ZNSt12_Destroy_auxILb0EE9__destroyIPN2cv3MatEEEvT_S5_>
  4060a1:	c9                   	leaveq 
  4060a2:	c3                   	retq   
  4060a3:	90                   	nop

00000000004060a4 <_ZNSt12_Vector_baseIfSaIfEE12_Vector_implC1Ev>:
  4060a4:	55                   	push   %rbp
  4060a5:	48 89 e5             	mov    %rsp,%rbp
  4060a8:	48 83 ec 10          	sub    $0x10,%rsp
  4060ac:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4060b0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4060b4:	48 89 c7             	mov    %rax,%rdi
  4060b7:	e8 dc 05 00 00       	callq  406698 <_ZNSaIfEC1Ev>
  4060bc:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4060c0:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  4060c7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4060cb:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  4060d2:	00 
  4060d3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4060d7:	48 c7 40 10 00 00 00 	movq   $0x0,0x10(%rax)
  4060de:	00 
  4060df:	c9                   	leaveq 
  4060e0:	c3                   	retq   
  4060e1:	90                   	nop

00000000004060e2 <_ZNSaIfED1Ev>:
  4060e2:	55                   	push   %rbp
  4060e3:	48 89 e5             	mov    %rsp,%rbp
  4060e6:	48 83 ec 10          	sub    $0x10,%rsp
  4060ea:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4060ee:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4060f2:	48 89 c7             	mov    %rax,%rdi
  4060f5:	e8 b8 05 00 00       	callq  4066b2 <_ZN9__gnu_cxx13new_allocatorIfED1Ev>
  4060fa:	c9                   	leaveq 
  4060fb:	c3                   	retq   

00000000004060fc <_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm>:
  4060fc:	55                   	push   %rbp
  4060fd:	48 89 e5             	mov    %rsp,%rbp
  406100:	48 83 ec 20          	sub    $0x20,%rsp
  406104:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406108:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40610c:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  406110:	48 83 7d f0 00       	cmpq   $0x0,-0x10(%rbp)
  406115:	74 17                	je     40612e <_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm+0x32>
  406117:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40611b:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  40611f:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  406123:	48 89 ce             	mov    %rcx,%rsi
  406126:	48 89 c7             	mov    %rax,%rdi
  406129:	e8 8e 05 00 00       	callq  4066bc <_ZN9__gnu_cxx13new_allocatorIfE10deallocateEPfm>
  40612e:	c9                   	leaveq 
  40612f:	c3                   	retq   

0000000000406130 <_ZSt8_DestroyIPfEvT_S1_>:
  406130:	55                   	push   %rbp
  406131:	48 89 e5             	mov    %rsp,%rbp
  406134:	48 83 ec 10          	sub    $0x10,%rsp
  406138:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40613c:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  406140:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  406144:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406148:	48 89 d6             	mov    %rdx,%rsi
  40614b:	48 89 c7             	mov    %rax,%rdi
  40614e:	e8 8b 05 00 00       	callq  4066de <_ZNSt12_Destroy_auxILb1EE9__destroyIPfEEvT_S3_>
  406153:	c9                   	leaveq 
  406154:	c3                   	retq   
  406155:	90                   	nop

0000000000406156 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag>:
  406156:	55                   	push   %rbp
  406157:	48 89 e5             	mov    %rsp,%rbp
  40615a:	48 83 ec 40          	sub    $0x40,%rsp
  40615e:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  406162:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  406166:	48 89 55 c8          	mov    %rdx,-0x38(%rbp)
  40616a:	48 8b 55 c8          	mov    -0x38(%rbp),%rdx
  40616e:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  406172:	48 89 d6             	mov    %rdx,%rsi
  406175:	48 89 c7             	mov    %rax,%rdi
  406178:	e8 6f 05 00 00       	callq  4066ec <_ZSt8distanceIPhENSt15iterator_traitsIT_E15difference_typeES2_S2_>
  40617d:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  406181:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  406185:	48 89 c7             	mov    %rax,%rdi
  406188:	e8 99 05 00 00       	callq  406726 <_ZNKSt6vectorIfSaIfEE8capacityEv>
  40618d:	48 3b 45 f8          	cmp    -0x8(%rbp),%rax
  406191:	0f 92 c0             	setb   %al
  406194:	84 c0                	test   %al,%al
  406196:	0f 84 b4 00 00 00    	je     406250 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0xfa>
  40619c:	48 8b 4d c8          	mov    -0x38(%rbp),%rcx
  4061a0:	48 8b 55 d0          	mov    -0x30(%rbp),%rdx
  4061a4:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  4061a8:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4061ac:	48 89 c7             	mov    %rax,%rdi
  4061af:	e8 98 05 00 00       	callq  40674c <_ZNSt6vectorIfSaIfEE20_M_allocate_and_copyIPhEEPfmT_S5_>
  4061b4:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  4061b8:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4061bc:	48 89 c7             	mov    %rax,%rdi
  4061bf:	e8 e6 f8 ff ff       	callq  405aaa <_ZNSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv>
  4061c4:	48 89 c2             	mov    %rax,%rdx
  4061c7:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4061cb:	48 8b 48 08          	mov    0x8(%rax),%rcx
  4061cf:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4061d3:	48 8b 00             	mov    (%rax),%rax
  4061d6:	48 89 ce             	mov    %rcx,%rsi
  4061d9:	48 89 c7             	mov    %rax,%rdi
  4061dc:	e8 d7 f8 ff ff       	callq  405ab8 <_ZSt8_DestroyIPffEvT_S1_RSaIT0_E>
  4061e1:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4061e5:	48 8b 40 10          	mov    0x10(%rax),%rax
  4061e9:	48 89 c2             	mov    %rax,%rdx
  4061ec:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4061f0:	48 8b 00             	mov    (%rax),%rax
  4061f3:	48 29 c2             	sub    %rax,%rdx
  4061f6:	48 89 d0             	mov    %rdx,%rax
  4061f9:	48 c1 f8 02          	sar    $0x2,%rax
  4061fd:	48 89 c2             	mov    %rax,%rdx
  406200:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  406204:	48 8b 08             	mov    (%rax),%rcx
  406207:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  40620b:	48 89 ce             	mov    %rcx,%rsi
  40620e:	48 89 c7             	mov    %rax,%rdi
  406211:	e8 e6 fe ff ff       	callq  4060fc <_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm>
  406216:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  40621a:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  40621e:	48 89 10             	mov    %rdx,(%rax)
  406221:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  406225:	48 8b 00             	mov    (%rax),%rax
  406228:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  40622c:	48 c1 e2 02          	shl    $0x2,%rdx
  406230:	48 01 c2             	add    %rax,%rdx
  406233:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  406237:	48 89 50 08          	mov    %rdx,0x8(%rax)
  40623b:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  40623f:	48 8b 50 08          	mov    0x8(%rax),%rdx
  406243:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  406247:	48 89 50 10          	mov    %rdx,0x10(%rax)
  40624b:	e9 b4 00 00 00       	jmpq   406304 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x1ae>
  406250:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  406254:	48 89 c7             	mov    %rax,%rdi
  406257:	e8 88 05 00 00       	callq  4067e4 <_ZNKSt6vectorIfSaIfEE4sizeEv>
  40625c:	48 3b 45 f8          	cmp    -0x8(%rbp),%rax
  406260:	0f 93 c0             	setae  %al
  406263:	84 c0                	test   %al,%al
  406265:	74 2e                	je     406295 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x13f>
  406267:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  40626b:	48 8b 10             	mov    (%rax),%rdx
  40626e:	48 8b 4d c8          	mov    -0x38(%rbp),%rcx
  406272:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  406276:	48 89 ce             	mov    %rcx,%rsi
  406279:	48 89 c7             	mov    %rax,%rdi
  40627c:	e8 cd 05 00 00       	callq  40684e <_ZSt4copyIPhPfET0_T_S3_S2_>
  406281:	48 89 c2             	mov    %rax,%rdx
  406284:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  406288:	48 89 d6             	mov    %rdx,%rsi
  40628b:	48 89 c7             	mov    %rax,%rdi
  40628e:	e8 77 05 00 00       	callq  40680a <_ZNSt6vectorIfSaIfEE15_M_erase_at_endEPf>
  406293:	eb 6f                	jmp    406304 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x1ae>
  406295:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  406299:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  40629d:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4062a1:	48 89 c7             	mov    %rax,%rdi
  4062a4:	e8 3b 05 00 00       	callq  4067e4 <_ZNKSt6vectorIfSaIfEE4sizeEv>
  4062a9:	48 89 c2             	mov    %rax,%rdx
  4062ac:	48 8d 45 e8          	lea    -0x18(%rbp),%rax
  4062b0:	48 89 d6             	mov    %rdx,%rsi
  4062b3:	48 89 c7             	mov    %rax,%rdi
  4062b6:	e8 d9 05 00 00       	callq  406894 <_ZSt7advanceIPhmEvRT_T0_>
  4062bb:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4062bf:	48 8b 10             	mov    (%rax),%rdx
  4062c2:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  4062c6:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  4062ca:	48 89 ce             	mov    %rcx,%rsi
  4062cd:	48 89 c7             	mov    %rax,%rdi
  4062d0:	e8 79 05 00 00       	callq  40684e <_ZSt4copyIPhPfET0_T_S3_S2_>
  4062d5:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4062d9:	48 89 c7             	mov    %rax,%rdi
  4062dc:	e8 c9 f7 ff ff       	callq  405aaa <_ZNSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv>
  4062e1:	48 89 c1             	mov    %rax,%rcx
  4062e4:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4062e8:	48 8b 50 08          	mov    0x8(%rax),%rdx
  4062ec:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4062f0:	48 8b 75 c8          	mov    -0x38(%rbp),%rsi
  4062f4:	48 89 c7             	mov    %rax,%rdi
  4062f7:	e8 da 05 00 00       	callq  4068d6 <_ZSt22__uninitialized_copy_aIPhPffET0_T_S3_S2_RSaIT1_E>
  4062fc:	48 8b 55 d8          	mov    -0x28(%rbp),%rdx
  406300:	48 89 42 08          	mov    %rax,0x8(%rdx)
  406304:	c9                   	leaveq 
  406305:	c3                   	retq   

0000000000406306 <_ZN9__gnu_cxx13new_allocatorIN2cv3MatEE9constructEPS2_RKS2_>:
  406306:	55                   	push   %rbp
  406307:	48 89 e5             	mov    %rsp,%rbp
  40630a:	41 55                	push   %r13
  40630c:	41 54                	push   %r12
  40630e:	53                   	push   %rbx
  40630f:	48 83 ec 28          	sub    $0x28,%rsp
  406313:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  406317:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  40631b:	48 89 55 c8          	mov    %rdx,-0x38(%rbp)
  40631f:	4c 8b 65 d0          	mov    -0x30(%rbp),%r12
  406323:	4c 89 e6             	mov    %r12,%rsi
  406326:	bf 60 00 00 00       	mov    $0x60,%edi
  40632b:	e8 fe ea ff ff       	callq  404e2e <_ZnwmPv>
  406330:	48 89 c3             	mov    %rax,%rbx
  406333:	48 85 db             	test   %rbx,%rbx
  406336:	74 11                	je     406349 <_ZN9__gnu_cxx13new_allocatorIN2cv3MatEE9constructEPS2_RKS2_+0x43>
  406338:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  40633c:	48 89 c6             	mov    %rax,%rsi
  40633f:	48 89 df             	mov    %rbx,%rdi
  406342:	e8 01 ec ff ff       	callq  404f48 <_ZN2cv3MatC1ERKS0_>
  406347:	eb 1b                	jmp    406364 <_ZN9__gnu_cxx13new_allocatorIN2cv3MatEE9constructEPS2_RKS2_+0x5e>
  406349:	eb 19                	jmp    406364 <_ZN9__gnu_cxx13new_allocatorIN2cv3MatEE9constructEPS2_RKS2_+0x5e>
  40634b:	49 89 c5             	mov    %rax,%r13
  40634e:	4c 89 e6             	mov    %r12,%rsi
  406351:	48 89 df             	mov    %rbx,%rdi
  406354:	e8 e7 ea ff ff       	callq  404e40 <_ZdlPvS_>
  406359:	4c 89 e8             	mov    %r13,%rax
  40635c:	48 89 c7             	mov    %rax,%rdi
  40635f:	e8 dc b4 ff ff       	callq  401840 <_Unwind_Resume@plt>
  406364:	48 83 c4 28          	add    $0x28,%rsp
  406368:	5b                   	pop    %rbx
  406369:	41 5c                	pop    %r12
  40636b:	41 5d                	pop    %r13
  40636d:	5d                   	pop    %rbp
  40636e:	c3                   	retq   
  40636f:	90                   	nop

0000000000406370 <_ZNK9__gnu_cxx17__normal_iteratorIPN2cv3MatESt6vectorIS2_SaIS2_EEE4baseEv>:
  406370:	55                   	push   %rbp
  406371:	48 89 e5             	mov    %rsp,%rbp
  406374:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406378:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40637c:	5d                   	pop    %rbp
  40637d:	c3                   	retq   

000000000040637e <_ZSt13copy_backwardIPN2cv3MatES2_ET0_T_S4_S3_>:
  40637e:	55                   	push   %rbp
  40637f:	48 89 e5             	mov    %rsp,%rbp
  406382:	53                   	push   %rbx
  406383:	48 83 ec 28          	sub    $0x28,%rsp
  406387:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40638b:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  40638f:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
  406393:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  406397:	48 89 c7             	mov    %rax,%rdi
  40639a:	e8 68 05 00 00       	callq  406907 <_ZSt12__miter_baseIPN2cv3MatEENSt11_Miter_baseIT_E13iterator_typeES4_>
  40639f:	48 89 c3             	mov    %rax,%rbx
  4063a2:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4063a6:	48 89 c7             	mov    %rax,%rdi
  4063a9:	e8 59 05 00 00       	callq  406907 <_ZSt12__miter_baseIPN2cv3MatEENSt11_Miter_baseIT_E13iterator_typeES4_>
  4063ae:	48 8b 55 d8          	mov    -0x28(%rbp),%rdx
  4063b2:	48 89 de             	mov    %rbx,%rsi
  4063b5:	48 89 c7             	mov    %rax,%rdi
  4063b8:	e8 64 05 00 00       	callq  406921 <_ZSt23__copy_move_backward_a2ILb0EPN2cv3MatES2_ET1_T0_S4_S3_>
  4063bd:	48 83 c4 28          	add    $0x28,%rsp
  4063c1:	5b                   	pop    %rbx
  4063c2:	5d                   	pop    %rbp
  4063c3:	c3                   	retq   

00000000004063c4 <_ZNK9__gnu_cxx17__normal_iteratorIPN2cv3MatESt6vectorIS2_SaIS2_EEEdeEv>:
  4063c4:	55                   	push   %rbp
  4063c5:	48 89 e5             	mov    %rsp,%rbp
  4063c8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4063cc:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4063d0:	48 8b 00             	mov    (%rax),%rax
  4063d3:	5d                   	pop    %rbp
  4063d4:	c3                   	retq   
  4063d5:	90                   	nop

00000000004063d6 <_ZNKSt6vectorIN2cv3MatESaIS1_EE12_M_check_lenEmPKc>:
  4063d6:	55                   	push   %rbp
  4063d7:	48 89 e5             	mov    %rsp,%rbp
  4063da:	53                   	push   %rbx
  4063db:	48 83 ec 38          	sub    $0x38,%rsp
  4063df:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  4063e3:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  4063e7:	48 89 55 c8          	mov    %rdx,-0x38(%rbp)
  4063eb:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4063ef:	48 89 c7             	mov    %rax,%rdi
  4063f2:	e8 bb 05 00 00       	callq  4069b2 <_ZNKSt6vectorIN2cv3MatESaIS1_EE8max_sizeEv>
  4063f7:	48 89 c3             	mov    %rax,%rbx
  4063fa:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4063fe:	48 89 c7             	mov    %rax,%rdi
  406401:	e8 74 05 00 00       	callq  40697a <_ZNKSt6vectorIN2cv3MatESaIS1_EE4sizeEv>
  406406:	48 29 c3             	sub    %rax,%rbx
  406409:	48 89 da             	mov    %rbx,%rdx
  40640c:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  406410:	48 39 c2             	cmp    %rax,%rdx
  406413:	0f 92 c0             	setb   %al
  406416:	84 c0                	test   %al,%al
  406418:	74 0c                	je     406426 <_ZNKSt6vectorIN2cv3MatESaIS1_EE12_M_check_lenEmPKc+0x50>
  40641a:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  40641e:	48 89 c7             	mov    %rax,%rdi
  406421:	e8 ea b3 ff ff       	callq  401810 <_ZSt20__throw_length_errorPKc@plt>
  406426:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  40642a:	48 89 c7             	mov    %rax,%rdi
  40642d:	e8 48 05 00 00       	callq  40697a <_ZNKSt6vectorIN2cv3MatESaIS1_EE4sizeEv>
  406432:	48 89 c3             	mov    %rax,%rbx
  406435:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  406439:	48 89 c7             	mov    %rax,%rdi
  40643c:	e8 39 05 00 00       	callq  40697a <_ZNKSt6vectorIN2cv3MatESaIS1_EE4sizeEv>
  406441:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  406445:	48 8d 55 d0          	lea    -0x30(%rbp),%rdx
  406449:	48 8d 45 e0          	lea    -0x20(%rbp),%rax
  40644d:	48 89 d6             	mov    %rdx,%rsi
  406450:	48 89 c7             	mov    %rax,%rdi
  406453:	e8 7c 05 00 00       	callq  4069d4 <_ZSt3maxImERKT_S2_S2_>
  406458:	48 8b 00             	mov    (%rax),%rax
  40645b:	48 01 d8             	add    %rbx,%rax
  40645e:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  406462:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  406466:	48 89 c7             	mov    %rax,%rdi
  406469:	e8 0c 05 00 00       	callq  40697a <_ZNKSt6vectorIN2cv3MatESaIS1_EE4sizeEv>
  40646e:	48 3b 45 e8          	cmp    -0x18(%rbp),%rax
  406472:	77 12                	ja     406486 <_ZNKSt6vectorIN2cv3MatESaIS1_EE12_M_check_lenEmPKc+0xb0>
  406474:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  406478:	48 89 c7             	mov    %rax,%rdi
  40647b:	e8 32 05 00 00       	callq  4069b2 <_ZNKSt6vectorIN2cv3MatESaIS1_EE8max_sizeEv>
  406480:	48 3b 45 e8          	cmp    -0x18(%rbp),%rax
  406484:	73 0e                	jae    406494 <_ZNKSt6vectorIN2cv3MatESaIS1_EE12_M_check_lenEmPKc+0xbe>
  406486:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  40648a:	48 89 c7             	mov    %rax,%rdi
  40648d:	e8 20 05 00 00       	callq  4069b2 <_ZNKSt6vectorIN2cv3MatESaIS1_EE8max_sizeEv>
  406492:	eb 04                	jmp    406498 <_ZNKSt6vectorIN2cv3MatESaIS1_EE12_M_check_lenEmPKc+0xc2>
  406494:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  406498:	48 83 c4 38          	add    $0x38,%rsp
  40649c:	5b                   	pop    %rbx
  40649d:	5d                   	pop    %rbp
  40649e:	c3                   	retq   
  40649f:	90                   	nop

00000000004064a0 <_ZNSt6vectorIN2cv3MatESaIS1_EE5beginEv>:
  4064a0:	55                   	push   %rbp
  4064a1:	48 89 e5             	mov    %rsp,%rbp
  4064a4:	48 83 ec 20          	sub    $0x20,%rsp
  4064a8:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4064ac:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  4064b0:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  4064b4:	48 89 d6             	mov    %rdx,%rsi
  4064b7:	48 89 c7             	mov    %rax,%rdi
  4064ba:	e8 eb 00 00 00       	callq  4065aa <_ZN9__gnu_cxx17__normal_iteratorIPN2cv3MatESt6vectorIS2_SaIS2_EEEC1ERKS3_>
  4064bf:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4064c3:	c9                   	leaveq 
  4064c4:	c3                   	retq   

00000000004064c5 <_ZN9__gnu_cxxmiIPN2cv3MatESt6vectorIS2_SaIS2_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKSA_SD_>:
  4064c5:	55                   	push   %rbp
  4064c6:	48 89 e5             	mov    %rsp,%rbp
  4064c9:	53                   	push   %rbx
  4064ca:	48 83 ec 18          	sub    $0x18,%rsp
  4064ce:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4064d2:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  4064d6:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4064da:	48 89 c7             	mov    %rax,%rdi
  4064dd:	e8 8e fe ff ff       	callq  406370 <_ZNK9__gnu_cxx17__normal_iteratorIPN2cv3MatESt6vectorIS2_SaIS2_EEE4baseEv>
  4064e2:	48 8b 00             	mov    (%rax),%rax
  4064e5:	48 89 c3             	mov    %rax,%rbx
  4064e8:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4064ec:	48 89 c7             	mov    %rax,%rdi
  4064ef:	e8 7c fe ff ff       	callq  406370 <_ZNK9__gnu_cxx17__normal_iteratorIPN2cv3MatESt6vectorIS2_SaIS2_EEE4baseEv>
  4064f4:	48 8b 00             	mov    (%rax),%rax
  4064f7:	48 29 c3             	sub    %rax,%rbx
  4064fa:	48 89 d8             	mov    %rbx,%rax
  4064fd:	48 c1 f8 05          	sar    $0x5,%rax
  406501:	48 89 c2             	mov    %rax,%rdx
  406504:	48 b8 ab aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaab,%rax
  40650b:	aa aa aa 
  40650e:	48 0f af c2          	imul   %rdx,%rax
  406512:	48 83 c4 18          	add    $0x18,%rsp
  406516:	5b                   	pop    %rbx
  406517:	5d                   	pop    %rbp
  406518:	c3                   	retq   
  406519:	90                   	nop

000000000040651a <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE11_M_allocateEm>:
  40651a:	55                   	push   %rbp
  40651b:	48 89 e5             	mov    %rsp,%rbp
  40651e:	48 83 ec 10          	sub    $0x10,%rsp
  406522:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406526:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40652a:	48 83 7d f0 00       	cmpq   $0x0,-0x10(%rbp)
  40652f:	74 1a                	je     40654b <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE11_M_allocateEm+0x31>
  406531:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406535:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  406539:	ba 00 00 00 00       	mov    $0x0,%edx
  40653e:	48 89 ce             	mov    %rcx,%rsi
  406541:	48 89 c7             	mov    %rax,%rdi
  406544:	e8 b7 04 00 00       	callq  406a00 <_ZN9__gnu_cxx13new_allocatorIN2cv3MatEE8allocateEmPKv>
  406549:	eb 05                	jmp    406550 <_ZNSt12_Vector_baseIN2cv3MatESaIS1_EE11_M_allocateEm+0x36>
  40654b:	b8 00 00 00 00       	mov    $0x0,%eax
  406550:	c9                   	leaveq 
  406551:	c3                   	retq   

0000000000406552 <_ZSt34__uninitialized_move_if_noexcept_aIPN2cv3MatES2_SaIS1_EET0_T_S5_S4_RT1_>:
  406552:	55                   	push   %rbp
  406553:	48 89 e5             	mov    %rsp,%rbp
  406556:	48 83 ec 20          	sub    $0x20,%rsp
  40655a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40655e:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  406562:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  406566:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  40656a:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  40656e:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  406572:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  406576:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40657a:	48 89 c7             	mov    %rax,%rdi
  40657d:	e8 c9 04 00 00       	callq  406a4b <_ZSt22__uninitialized_copy_aIPN2cv3MatES2_S1_ET0_T_S4_S3_RSaIT1_E>
  406582:	c9                   	leaveq 
  406583:	c3                   	retq   

0000000000406584 <_ZN9__gnu_cxx14__alloc_traitsISaIN2cv3MatEEE7destroyERS3_PS2_>:
  406584:	55                   	push   %rbp
  406585:	48 89 e5             	mov    %rsp,%rbp
  406588:	48 83 ec 10          	sub    $0x10,%rsp
  40658c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406590:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  406594:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  406598:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40659c:	48 89 d6             	mov    %rdx,%rsi
  40659f:	48 89 c7             	mov    %rax,%rdi
  4065a2:	e8 d5 04 00 00       	callq  406a7c <_ZN9__gnu_cxx13new_allocatorIN2cv3MatEE7destroyEPS2_>
  4065a7:	c9                   	leaveq 
  4065a8:	c3                   	retq   
  4065a9:	90                   	nop

00000000004065aa <_ZN9__gnu_cxx17__normal_iteratorIPN2cv3MatESt6vectorIS2_SaIS2_EEEC1ERKS3_>:
  4065aa:	55                   	push   %rbp
  4065ab:	48 89 e5             	mov    %rsp,%rbp
  4065ae:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4065b2:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4065b6:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4065ba:	48 8b 10             	mov    (%rax),%rdx
  4065bd:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4065c1:	48 89 10             	mov    %rdx,(%rax)
  4065c4:	5d                   	pop    %rbp
  4065c5:	c3                   	retq   

00000000004065c6 <_ZNSaIiEC1Ev>:
  4065c6:	55                   	push   %rbp
  4065c7:	48 89 e5             	mov    %rsp,%rbp
  4065ca:	48 83 ec 10          	sub    $0x10,%rsp
  4065ce:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4065d2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4065d6:	48 89 c7             	mov    %rax,%rdi
  4065d9:	e8 bc 04 00 00       	callq  406a9a <_ZN9__gnu_cxx13new_allocatorIiEC1Ev>
  4065de:	c9                   	leaveq 
  4065df:	c3                   	retq   

00000000004065e0 <_ZN9__gnu_cxx13new_allocatorIiED1Ev>:
  4065e0:	55                   	push   %rbp
  4065e1:	48 89 e5             	mov    %rsp,%rbp
  4065e4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4065e8:	5d                   	pop    %rbp
  4065e9:	c3                   	retq   

00000000004065ea <_ZN9__gnu_cxx13new_allocatorIiE10deallocateEPim>:
  4065ea:	55                   	push   %rbp
  4065eb:	48 89 e5             	mov    %rsp,%rbp
  4065ee:	48 83 ec 20          	sub    $0x20,%rsp
  4065f2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4065f6:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4065fa:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  4065fe:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  406602:	48 89 c7             	mov    %rax,%rdi
  406605:	e8 c6 b1 ff ff       	callq  4017d0 <_ZdlPv@plt>
  40660a:	c9                   	leaveq 
  40660b:	c3                   	retq   

000000000040660c <_ZNSt12_Destroy_auxILb1EE9__destroyIPiEEvT_S3_>:
  40660c:	55                   	push   %rbp
  40660d:	48 89 e5             	mov    %rsp,%rbp
  406610:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406614:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  406618:	5d                   	pop    %rbp
  406619:	c3                   	retq   

000000000040661a <_ZNSaIN2cv3MatEEC1Ev>:
  40661a:	55                   	push   %rbp
  40661b:	48 89 e5             	mov    %rsp,%rbp
  40661e:	48 83 ec 10          	sub    $0x10,%rsp
  406622:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406626:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40662a:	48 89 c7             	mov    %rax,%rdi
  40662d:	e8 72 04 00 00       	callq  406aa4 <_ZN9__gnu_cxx13new_allocatorIN2cv3MatEEC1Ev>
  406632:	c9                   	leaveq 
  406633:	c3                   	retq   

0000000000406634 <_ZN9__gnu_cxx13new_allocatorIN2cv3MatEED1Ev>:
  406634:	55                   	push   %rbp
  406635:	48 89 e5             	mov    %rsp,%rbp
  406638:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40663c:	5d                   	pop    %rbp
  40663d:	c3                   	retq   

000000000040663e <_ZN9__gnu_cxx13new_allocatorIN2cv3MatEE10deallocateEPS2_m>:
  40663e:	55                   	push   %rbp
  40663f:	48 89 e5             	mov    %rsp,%rbp
  406642:	48 83 ec 20          	sub    $0x20,%rsp
  406646:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40664a:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40664e:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  406652:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  406656:	48 89 c7             	mov    %rax,%rdi
  406659:	e8 72 b1 ff ff       	callq  4017d0 <_ZdlPv@plt>
  40665e:	c9                   	leaveq 
  40665f:	c3                   	retq   

0000000000406660 <_ZNSt12_Destroy_auxILb0EE9__destroyIPN2cv3MatEEEvT_S5_>:
  406660:	55                   	push   %rbp
  406661:	48 89 e5             	mov    %rsp,%rbp
  406664:	48 83 ec 10          	sub    $0x10,%rsp
  406668:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40666c:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  406670:	eb 19                	jmp    40668b <_ZNSt12_Destroy_auxILb0EE9__destroyIPN2cv3MatEEEvT_S5_+0x2b>
  406672:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406676:	48 89 c7             	mov    %rax,%rdi
  406679:	e8 30 04 00 00       	callq  406aae <_ZSt11__addressofIN2cv3MatEEPT_RS2_>
  40667e:	48 89 c7             	mov    %rax,%rdi
  406681:	e8 36 04 00 00       	callq  406abc <_ZSt8_DestroyIN2cv3MatEEvPT_>
  406686:	48 83 45 f8 60       	addq   $0x60,-0x8(%rbp)
  40668b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40668f:	48 3b 45 f0          	cmp    -0x10(%rbp),%rax
  406693:	75 dd                	jne    406672 <_ZNSt12_Destroy_auxILb0EE9__destroyIPN2cv3MatEEEvT_S5_+0x12>
  406695:	c9                   	leaveq 
  406696:	c3                   	retq   
  406697:	90                   	nop

0000000000406698 <_ZNSaIfEC1Ev>:
  406698:	55                   	push   %rbp
  406699:	48 89 e5             	mov    %rsp,%rbp
  40669c:	48 83 ec 10          	sub    $0x10,%rsp
  4066a0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4066a4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4066a8:	48 89 c7             	mov    %rax,%rdi
  4066ab:	e8 26 04 00 00       	callq  406ad6 <_ZN9__gnu_cxx13new_allocatorIfEC1Ev>
  4066b0:	c9                   	leaveq 
  4066b1:	c3                   	retq   

00000000004066b2 <_ZN9__gnu_cxx13new_allocatorIfED1Ev>:
  4066b2:	55                   	push   %rbp
  4066b3:	48 89 e5             	mov    %rsp,%rbp
  4066b6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4066ba:	5d                   	pop    %rbp
  4066bb:	c3                   	retq   

00000000004066bc <_ZN9__gnu_cxx13new_allocatorIfE10deallocateEPfm>:
  4066bc:	55                   	push   %rbp
  4066bd:	48 89 e5             	mov    %rsp,%rbp
  4066c0:	48 83 ec 20          	sub    $0x20,%rsp
  4066c4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4066c8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4066cc:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  4066d0:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4066d4:	48 89 c7             	mov    %rax,%rdi
  4066d7:	e8 f4 b0 ff ff       	callq  4017d0 <_ZdlPv@plt>
  4066dc:	c9                   	leaveq 
  4066dd:	c3                   	retq   

00000000004066de <_ZNSt12_Destroy_auxILb1EE9__destroyIPfEEvT_S3_>:
  4066de:	55                   	push   %rbp
  4066df:	48 89 e5             	mov    %rsp,%rbp
  4066e2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4066e6:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4066ea:	5d                   	pop    %rbp
  4066eb:	c3                   	retq   

00000000004066ec <_ZSt8distanceIPhENSt15iterator_traitsIT_E15difference_typeES2_S2_>:
  4066ec:	55                   	push   %rbp
  4066ed:	48 89 e5             	mov    %rsp,%rbp
  4066f0:	53                   	push   %rbx
  4066f1:	48 83 ec 38          	sub    $0x38,%rsp
  4066f5:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  4066f9:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  4066fd:	48 8d 45 d8          	lea    -0x28(%rbp),%rax
  406701:	48 89 c7             	mov    %rax,%rdi
  406704:	e8 d7 03 00 00       	callq  406ae0 <_ZSt19__iterator_categoryIPhENSt15iterator_traitsIT_E17iterator_categoryERKS2_>
  406709:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  40670d:	48 8b 55 d0          	mov    -0x30(%rbp),%rdx
  406711:	88 1c 24             	mov    %bl,(%rsp)
  406714:	48 89 d6             	mov    %rdx,%rsi
  406717:	48 89 c7             	mov    %rax,%rdi
  40671a:	e8 cb 03 00 00       	callq  406aea <_ZSt10__distanceIPhENSt15iterator_traitsIT_E15difference_typeES2_S2_St26random_access_iterator_tag>
  40671f:	48 83 c4 38          	add    $0x38,%rsp
  406723:	5b                   	pop    %rbx
  406724:	5d                   	pop    %rbp
  406725:	c3                   	retq   

0000000000406726 <_ZNKSt6vectorIfSaIfEE8capacityEv>:
  406726:	55                   	push   %rbp
  406727:	48 89 e5             	mov    %rsp,%rbp
  40672a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40672e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406732:	48 8b 40 10          	mov    0x10(%rax),%rax
  406736:	48 89 c2             	mov    %rax,%rdx
  406739:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40673d:	48 8b 00             	mov    (%rax),%rax
  406740:	48 29 c2             	sub    %rax,%rdx
  406743:	48 89 d0             	mov    %rdx,%rax
  406746:	48 c1 f8 02          	sar    $0x2,%rax
  40674a:	5d                   	pop    %rbp
  40674b:	c3                   	retq   

000000000040674c <_ZNSt6vectorIfSaIfEE20_M_allocate_and_copyIPhEEPfmT_S5_>:
  40674c:	55                   	push   %rbp
  40674d:	48 89 e5             	mov    %rsp,%rbp
  406750:	53                   	push   %rbx
  406751:	48 83 ec 38          	sub    $0x38,%rsp
  406755:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  406759:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  40675d:	48 89 55 c8          	mov    %rdx,-0x38(%rbp)
  406761:	48 89 4d c0          	mov    %rcx,-0x40(%rbp)
  406765:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  406769:	48 8b 55 d0          	mov    -0x30(%rbp),%rdx
  40676d:	48 89 d6             	mov    %rdx,%rsi
  406770:	48 89 c7             	mov    %rax,%rdi
  406773:	e8 8e 03 00 00       	callq  406b06 <_ZNSt12_Vector_baseIfSaIfEE11_M_allocateEm>
  406778:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  40677c:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  406780:	48 89 c7             	mov    %rax,%rdi
  406783:	e8 22 f3 ff ff       	callq  405aaa <_ZNSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv>
  406788:	48 89 c1             	mov    %rax,%rcx
  40678b:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  40678f:	48 8b 75 c0          	mov    -0x40(%rbp),%rsi
  406793:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  406797:	48 89 c7             	mov    %rax,%rdi
  40679a:	e8 37 01 00 00       	callq  4068d6 <_ZSt22__uninitialized_copy_aIPhPffET0_T_S3_S2_RSaIT1_E>
  40679f:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4067a3:	eb 37                	jmp    4067dc <_ZNSt6vectorIfSaIfEE20_M_allocate_and_copyIPhEEPfmT_S5_+0x90>
  4067a5:	48 89 c7             	mov    %rax,%rdi
  4067a8:	e8 53 af ff ff       	callq  401700 <__cxa_begin_catch@plt>
  4067ad:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4067b1:	48 8b 55 d0          	mov    -0x30(%rbp),%rdx
  4067b5:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  4067b9:	48 89 ce             	mov    %rcx,%rsi
  4067bc:	48 89 c7             	mov    %rax,%rdi
  4067bf:	e8 38 f9 ff ff       	callq  4060fc <_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm>
  4067c4:	e8 e7 b0 ff ff       	callq  4018b0 <__cxa_rethrow@plt>
  4067c9:	48 89 c3             	mov    %rax,%rbx
  4067cc:	e8 bf b0 ff ff       	callq  401890 <__cxa_end_catch@plt>
  4067d1:	48 89 d8             	mov    %rbx,%rax
  4067d4:	48 89 c7             	mov    %rax,%rdi
  4067d7:	e8 64 b0 ff ff       	callq  401840 <_Unwind_Resume@plt>
  4067dc:	48 83 c4 38          	add    $0x38,%rsp
  4067e0:	5b                   	pop    %rbx
  4067e1:	5d                   	pop    %rbp
  4067e2:	c3                   	retq   
  4067e3:	90                   	nop

00000000004067e4 <_ZNKSt6vectorIfSaIfEE4sizeEv>:
  4067e4:	55                   	push   %rbp
  4067e5:	48 89 e5             	mov    %rsp,%rbp
  4067e8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4067ec:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4067f0:	48 8b 40 08          	mov    0x8(%rax),%rax
  4067f4:	48 89 c2             	mov    %rax,%rdx
  4067f7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4067fb:	48 8b 00             	mov    (%rax),%rax
  4067fe:	48 29 c2             	sub    %rax,%rdx
  406801:	48 89 d0             	mov    %rdx,%rax
  406804:	48 c1 f8 02          	sar    $0x2,%rax
  406808:	5d                   	pop    %rbp
  406809:	c3                   	retq   

000000000040680a <_ZNSt6vectorIfSaIfEE15_M_erase_at_endEPf>:
  40680a:	55                   	push   %rbp
  40680b:	48 89 e5             	mov    %rsp,%rbp
  40680e:	48 83 ec 10          	sub    $0x10,%rsp
  406812:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406816:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40681a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40681e:	48 89 c7             	mov    %rax,%rdi
  406821:	e8 84 f2 ff ff       	callq  405aaa <_ZNSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv>
  406826:	48 89 c2             	mov    %rax,%rdx
  406829:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40682d:	48 8b 48 08          	mov    0x8(%rax),%rcx
  406831:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  406835:	48 89 ce             	mov    %rcx,%rsi
  406838:	48 89 c7             	mov    %rax,%rdi
  40683b:	e8 78 f2 ff ff       	callq  405ab8 <_ZSt8_DestroyIPffEvT_S1_RSaIT0_E>
  406840:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406844:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  406848:	48 89 50 08          	mov    %rdx,0x8(%rax)
  40684c:	c9                   	leaveq 
  40684d:	c3                   	retq   

000000000040684e <_ZSt4copyIPhPfET0_T_S3_S2_>:
  40684e:	55                   	push   %rbp
  40684f:	48 89 e5             	mov    %rsp,%rbp
  406852:	53                   	push   %rbx
  406853:	48 83 ec 28          	sub    $0x28,%rsp
  406857:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40685b:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  40685f:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
  406863:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  406867:	48 89 c7             	mov    %rax,%rdi
  40686a:	e8 cf 02 00 00       	callq  406b3e <_ZSt12__miter_baseIPhENSt11_Miter_baseIT_E13iterator_typeES2_>
  40686f:	48 89 c3             	mov    %rax,%rbx
  406872:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  406876:	48 89 c7             	mov    %rax,%rdi
  406879:	e8 c0 02 00 00       	callq  406b3e <_ZSt12__miter_baseIPhENSt11_Miter_baseIT_E13iterator_typeES2_>
  40687e:	48 8b 55 d8          	mov    -0x28(%rbp),%rdx
  406882:	48 89 de             	mov    %rbx,%rsi
  406885:	48 89 c7             	mov    %rax,%rdi
  406888:	e8 cb 02 00 00       	callq  406b58 <_ZSt14__copy_move_a2ILb0EPhPfET1_T0_S3_S2_>
  40688d:	48 83 c4 28          	add    $0x28,%rsp
  406891:	5b                   	pop    %rbx
  406892:	5d                   	pop    %rbp
  406893:	c3                   	retq   

0000000000406894 <_ZSt7advanceIPhmEvRT_T0_>:
  406894:	55                   	push   %rbp
  406895:	48 89 e5             	mov    %rsp,%rbp
  406898:	53                   	push   %rbx
  406899:	48 83 ec 38          	sub    $0x38,%rsp
  40689d:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  4068a1:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  4068a5:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  4068a9:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4068ad:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4068b1:	48 89 c7             	mov    %rax,%rdi
  4068b4:	e8 27 02 00 00       	callq  406ae0 <_ZSt19__iterator_categoryIPhENSt15iterator_traitsIT_E17iterator_categoryERKS2_>
  4068b9:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  4068bd:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4068c1:	88 1c 24             	mov    %bl,(%rsp)
  4068c4:	48 89 d6             	mov    %rdx,%rsi
  4068c7:	48 89 c7             	mov    %rax,%rdi
  4068ca:	e8 e1 02 00 00       	callq  406bb0 <_ZSt9__advanceIPhlEvRT_T0_St26random_access_iterator_tag>
  4068cf:	48 83 c4 38          	add    $0x38,%rsp
  4068d3:	5b                   	pop    %rbx
  4068d4:	5d                   	pop    %rbp
  4068d5:	c3                   	retq   

00000000004068d6 <_ZSt22__uninitialized_copy_aIPhPffET0_T_S3_S2_RSaIT1_E>:
  4068d6:	55                   	push   %rbp
  4068d7:	48 89 e5             	mov    %rsp,%rbp
  4068da:	48 83 ec 20          	sub    $0x20,%rsp
  4068de:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4068e2:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4068e6:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  4068ea:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  4068ee:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  4068f2:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  4068f6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4068fa:	48 89 ce             	mov    %rcx,%rsi
  4068fd:	48 89 c7             	mov    %rax,%rdi
  406900:	e8 ce 02 00 00       	callq  406bd3 <_ZSt18uninitialized_copyIPhPfET0_T_S3_S2_>
  406905:	c9                   	leaveq 
  406906:	c3                   	retq   

0000000000406907 <_ZSt12__miter_baseIPN2cv3MatEENSt11_Miter_baseIT_E13iterator_typeES4_>:
  406907:	55                   	push   %rbp
  406908:	48 89 e5             	mov    %rsp,%rbp
  40690b:	48 83 ec 10          	sub    $0x10,%rsp
  40690f:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406913:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406917:	48 89 c7             	mov    %rax,%rdi
  40691a:	e8 e1 02 00 00       	callq  406c00 <_ZNSt10_Iter_baseIPN2cv3MatELb0EE7_S_baseES2_>
  40691f:	c9                   	leaveq 
  406920:	c3                   	retq   

0000000000406921 <_ZSt23__copy_move_backward_a2ILb0EPN2cv3MatES2_ET1_T0_S4_S3_>:
  406921:	55                   	push   %rbp
  406922:	48 89 e5             	mov    %rsp,%rbp
  406925:	41 54                	push   %r12
  406927:	53                   	push   %rbx
  406928:	48 83 ec 20          	sub    $0x20,%rsp
  40692c:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  406930:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  406934:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
  406938:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  40693c:	48 89 c7             	mov    %rax,%rdi
  40693f:	e8 ca 02 00 00       	callq  406c0e <_ZSt12__niter_baseIPN2cv3MatEENSt11_Niter_baseIT_E13iterator_typeES4_>
  406944:	49 89 c4             	mov    %rax,%r12
  406947:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40694b:	48 89 c7             	mov    %rax,%rdi
  40694e:	e8 bb 02 00 00       	callq  406c0e <_ZSt12__niter_baseIPN2cv3MatEENSt11_Niter_baseIT_E13iterator_typeES4_>
  406953:	48 89 c3             	mov    %rax,%rbx
  406956:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40695a:	48 89 c7             	mov    %rax,%rdi
  40695d:	e8 ac 02 00 00       	callq  406c0e <_ZSt12__niter_baseIPN2cv3MatEENSt11_Niter_baseIT_E13iterator_typeES4_>
  406962:	4c 89 e2             	mov    %r12,%rdx
  406965:	48 89 de             	mov    %rbx,%rsi
  406968:	48 89 c7             	mov    %rax,%rdi
  40696b:	e8 b8 02 00 00       	callq  406c28 <_ZSt22__copy_move_backward_aILb0EPN2cv3MatES2_ET1_T0_S4_S3_>
  406970:	48 83 c4 20          	add    $0x20,%rsp
  406974:	5b                   	pop    %rbx
  406975:	41 5c                	pop    %r12
  406977:	5d                   	pop    %rbp
  406978:	c3                   	retq   
  406979:	90                   	nop

000000000040697a <_ZNKSt6vectorIN2cv3MatESaIS1_EE4sizeEv>:
  40697a:	55                   	push   %rbp
  40697b:	48 89 e5             	mov    %rsp,%rbp
  40697e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406982:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406986:	48 8b 40 08          	mov    0x8(%rax),%rax
  40698a:	48 89 c2             	mov    %rax,%rdx
  40698d:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406991:	48 8b 00             	mov    (%rax),%rax
  406994:	48 29 c2             	sub    %rax,%rdx
  406997:	48 89 d0             	mov    %rdx,%rax
  40699a:	48 c1 f8 05          	sar    $0x5,%rax
  40699e:	48 89 c2             	mov    %rax,%rdx
  4069a1:	48 b8 ab aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaab,%rax
  4069a8:	aa aa aa 
  4069ab:	48 0f af c2          	imul   %rdx,%rax
  4069af:	5d                   	pop    %rbp
  4069b0:	c3                   	retq   
  4069b1:	90                   	nop

00000000004069b2 <_ZNKSt6vectorIN2cv3MatESaIS1_EE8max_sizeEv>:
  4069b2:	55                   	push   %rbp
  4069b3:	48 89 e5             	mov    %rsp,%rbp
  4069b6:	48 83 ec 10          	sub    $0x10,%rsp
  4069ba:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4069be:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4069c2:	48 89 c7             	mov    %rax,%rdi
  4069c5:	e8 aa 02 00 00       	callq  406c74 <_ZNKSt12_Vector_baseIN2cv3MatESaIS1_EE19_M_get_Tp_allocatorEv>
  4069ca:	48 89 c7             	mov    %rax,%rdi
  4069cd:	e8 87 02 00 00       	callq  406c59 <_ZN9__gnu_cxx14__alloc_traitsISaIN2cv3MatEEE8max_sizeERKS3_>
  4069d2:	c9                   	leaveq 
  4069d3:	c3                   	retq   

00000000004069d4 <_ZSt3maxImERKT_S2_S2_>:
  4069d4:	55                   	push   %rbp
  4069d5:	48 89 e5             	mov    %rsp,%rbp
  4069d8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4069dc:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4069e0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4069e4:	48 8b 10             	mov    (%rax),%rdx
  4069e7:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4069eb:	48 8b 00             	mov    (%rax),%rax
  4069ee:	48 39 c2             	cmp    %rax,%rdx
  4069f1:	73 06                	jae    4069f9 <_ZSt3maxImERKT_S2_S2_+0x25>
  4069f3:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4069f7:	eb 04                	jmp    4069fd <_ZSt3maxImERKT_S2_S2_+0x29>
  4069f9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4069fd:	5d                   	pop    %rbp
  4069fe:	c3                   	retq   
  4069ff:	90                   	nop

0000000000406a00 <_ZN9__gnu_cxx13new_allocatorIN2cv3MatEE8allocateEmPKv>:
  406a00:	55                   	push   %rbp
  406a01:	48 89 e5             	mov    %rsp,%rbp
  406a04:	48 83 ec 20          	sub    $0x20,%rsp
  406a08:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406a0c:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  406a10:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  406a14:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406a18:	48 89 c7             	mov    %rax,%rdi
  406a1b:	e8 62 02 00 00       	callq  406c82 <_ZNK9__gnu_cxx13new_allocatorIN2cv3MatEE8max_sizeEv>
  406a20:	48 3b 45 f0          	cmp    -0x10(%rbp),%rax
  406a24:	0f 92 c0             	setb   %al
  406a27:	84 c0                	test   %al,%al
  406a29:	74 05                	je     406a30 <_ZN9__gnu_cxx13new_allocatorIN2cv3MatEE8allocateEmPKv+0x30>
  406a2b:	e8 10 ad ff ff       	callq  401740 <_ZSt17__throw_bad_allocv@plt>
  406a30:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  406a34:	48 89 d0             	mov    %rdx,%rax
  406a37:	48 01 c0             	add    %rax,%rax
  406a3a:	48 01 d0             	add    %rdx,%rax
  406a3d:	48 c1 e0 05          	shl    $0x5,%rax
  406a41:	48 89 c7             	mov    %rax,%rdi
  406a44:	e8 17 ad ff ff       	callq  401760 <_Znwm@plt>
  406a49:	c9                   	leaveq 
  406a4a:	c3                   	retq   

0000000000406a4b <_ZSt22__uninitialized_copy_aIPN2cv3MatES2_S1_ET0_T_S4_S3_RSaIT1_E>:
  406a4b:	55                   	push   %rbp
  406a4c:	48 89 e5             	mov    %rsp,%rbp
  406a4f:	48 83 ec 20          	sub    $0x20,%rsp
  406a53:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406a57:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  406a5b:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  406a5f:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  406a63:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  406a67:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  406a6b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406a6f:	48 89 ce             	mov    %rcx,%rsi
  406a72:	48 89 c7             	mov    %rax,%rdi
  406a75:	e8 1c 02 00 00       	callq  406c96 <_ZSt18uninitialized_copyIPN2cv3MatES2_ET0_T_S4_S3_>
  406a7a:	c9                   	leaveq 
  406a7b:	c3                   	retq   

0000000000406a7c <_ZN9__gnu_cxx13new_allocatorIN2cv3MatEE7destroyEPS2_>:
  406a7c:	55                   	push   %rbp
  406a7d:	48 89 e5             	mov    %rsp,%rbp
  406a80:	48 83 ec 10          	sub    $0x10,%rsp
  406a84:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406a88:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  406a8c:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  406a90:	48 89 c7             	mov    %rax,%rdi
  406a93:	e8 24 e8 ff ff       	callq  4052bc <_ZN2cv3MatD1Ev>
  406a98:	c9                   	leaveq 
  406a99:	c3                   	retq   

0000000000406a9a <_ZN9__gnu_cxx13new_allocatorIiEC1Ev>:
  406a9a:	55                   	push   %rbp
  406a9b:	48 89 e5             	mov    %rsp,%rbp
  406a9e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406aa2:	5d                   	pop    %rbp
  406aa3:	c3                   	retq   

0000000000406aa4 <_ZN9__gnu_cxx13new_allocatorIN2cv3MatEEC1Ev>:
  406aa4:	55                   	push   %rbp
  406aa5:	48 89 e5             	mov    %rsp,%rbp
  406aa8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406aac:	5d                   	pop    %rbp
  406aad:	c3                   	retq   

0000000000406aae <_ZSt11__addressofIN2cv3MatEEPT_RS2_>:
  406aae:	55                   	push   %rbp
  406aaf:	48 89 e5             	mov    %rsp,%rbp
  406ab2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406ab6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406aba:	5d                   	pop    %rbp
  406abb:	c3                   	retq   

0000000000406abc <_ZSt8_DestroyIN2cv3MatEEvPT_>:
  406abc:	55                   	push   %rbp
  406abd:	48 89 e5             	mov    %rsp,%rbp
  406ac0:	48 83 ec 10          	sub    $0x10,%rsp
  406ac4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406ac8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406acc:	48 89 c7             	mov    %rax,%rdi
  406acf:	e8 e8 e7 ff ff       	callq  4052bc <_ZN2cv3MatD1Ev>
  406ad4:	c9                   	leaveq 
  406ad5:	c3                   	retq   

0000000000406ad6 <_ZN9__gnu_cxx13new_allocatorIfEC1Ev>:
  406ad6:	55                   	push   %rbp
  406ad7:	48 89 e5             	mov    %rsp,%rbp
  406ada:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406ade:	5d                   	pop    %rbp
  406adf:	c3                   	retq   

0000000000406ae0 <_ZSt19__iterator_categoryIPhENSt15iterator_traitsIT_E17iterator_categoryERKS2_>:
  406ae0:	55                   	push   %rbp
  406ae1:	48 89 e5             	mov    %rsp,%rbp
  406ae4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406ae8:	5d                   	pop    %rbp
  406ae9:	c3                   	retq   

0000000000406aea <_ZSt10__distanceIPhENSt15iterator_traitsIT_E15difference_typeES2_S2_St26random_access_iterator_tag>:
  406aea:	55                   	push   %rbp
  406aeb:	48 89 e5             	mov    %rsp,%rbp
  406aee:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406af2:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  406af6:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  406afa:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406afe:	48 29 c2             	sub    %rax,%rdx
  406b01:	48 89 d0             	mov    %rdx,%rax
  406b04:	5d                   	pop    %rbp
  406b05:	c3                   	retq   

0000000000406b06 <_ZNSt12_Vector_baseIfSaIfEE11_M_allocateEm>:
  406b06:	55                   	push   %rbp
  406b07:	48 89 e5             	mov    %rsp,%rbp
  406b0a:	48 83 ec 10          	sub    $0x10,%rsp
  406b0e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406b12:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  406b16:	48 83 7d f0 00       	cmpq   $0x0,-0x10(%rbp)
  406b1b:	74 1a                	je     406b37 <_ZNSt12_Vector_baseIfSaIfEE11_M_allocateEm+0x31>
  406b1d:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406b21:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  406b25:	ba 00 00 00 00       	mov    $0x0,%edx
  406b2a:	48 89 ce             	mov    %rcx,%rsi
  406b2d:	48 89 c7             	mov    %rax,%rdi
  406b30:	e8 8f 01 00 00       	callq  406cc4 <_ZN9__gnu_cxx13new_allocatorIfE8allocateEmPKv>
  406b35:	eb 05                	jmp    406b3c <_ZNSt12_Vector_baseIfSaIfEE11_M_allocateEm+0x36>
  406b37:	b8 00 00 00 00       	mov    $0x0,%eax
  406b3c:	c9                   	leaveq 
  406b3d:	c3                   	retq   

0000000000406b3e <_ZSt12__miter_baseIPhENSt11_Miter_baseIT_E13iterator_typeES2_>:
  406b3e:	55                   	push   %rbp
  406b3f:	48 89 e5             	mov    %rsp,%rbp
  406b42:	48 83 ec 10          	sub    $0x10,%rsp
  406b46:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406b4a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406b4e:	48 89 c7             	mov    %rax,%rdi
  406b51:	e8 b0 01 00 00       	callq  406d06 <_ZNSt10_Iter_baseIPhLb0EE7_S_baseES0_>
  406b56:	c9                   	leaveq 
  406b57:	c3                   	retq   

0000000000406b58 <_ZSt14__copy_move_a2ILb0EPhPfET1_T0_S3_S2_>:
  406b58:	55                   	push   %rbp
  406b59:	48 89 e5             	mov    %rsp,%rbp
  406b5c:	41 54                	push   %r12
  406b5e:	53                   	push   %rbx
  406b5f:	48 83 ec 20          	sub    $0x20,%rsp
  406b63:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  406b67:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  406b6b:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
  406b6f:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  406b73:	48 89 c7             	mov    %rax,%rdi
  406b76:	e8 b3 01 00 00       	callq  406d2e <_ZSt12__niter_baseIPfENSt11_Niter_baseIT_E13iterator_typeES2_>
  406b7b:	49 89 c4             	mov    %rax,%r12
  406b7e:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  406b82:	48 89 c7             	mov    %rax,%rdi
  406b85:	e8 8a 01 00 00       	callq  406d14 <_ZSt12__niter_baseIPhENSt11_Niter_baseIT_E13iterator_typeES2_>
  406b8a:	48 89 c3             	mov    %rax,%rbx
  406b8d:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  406b91:	48 89 c7             	mov    %rax,%rdi
  406b94:	e8 7b 01 00 00       	callq  406d14 <_ZSt12__niter_baseIPhENSt11_Niter_baseIT_E13iterator_typeES2_>
  406b99:	4c 89 e2             	mov    %r12,%rdx
  406b9c:	48 89 de             	mov    %rbx,%rsi
  406b9f:	48 89 c7             	mov    %rax,%rdi
  406ba2:	e8 a1 01 00 00       	callq  406d48 <_ZSt13__copy_move_aILb0EPhPfET1_T0_S3_S2_>
  406ba7:	48 83 c4 20          	add    $0x20,%rsp
  406bab:	5b                   	pop    %rbx
  406bac:	41 5c                	pop    %r12
  406bae:	5d                   	pop    %rbp
  406baf:	c3                   	retq   

0000000000406bb0 <_ZSt9__advanceIPhlEvRT_T0_St26random_access_iterator_tag>:
  406bb0:	55                   	push   %rbp
  406bb1:	48 89 e5             	mov    %rsp,%rbp
  406bb4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406bb8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  406bbc:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406bc0:	48 8b 10             	mov    (%rax),%rdx
  406bc3:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  406bc7:	48 01 c2             	add    %rax,%rdx
  406bca:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406bce:	48 89 10             	mov    %rdx,(%rax)
  406bd1:	5d                   	pop    %rbp
  406bd2:	c3                   	retq   

0000000000406bd3 <_ZSt18uninitialized_copyIPhPfET0_T_S3_S2_>:
  406bd3:	55                   	push   %rbp
  406bd4:	48 89 e5             	mov    %rsp,%rbp
  406bd7:	48 83 ec 20          	sub    $0x20,%rsp
  406bdb:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406bdf:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  406be3:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  406be7:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  406beb:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  406bef:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406bf3:	48 89 ce             	mov    %rcx,%rsi
  406bf6:	48 89 c7             	mov    %rax,%rdi
  406bf9:	e8 7b 01 00 00       	callq  406d79 <_ZNSt20__uninitialized_copyILb1EE13__uninit_copyIPhPfEET0_T_S5_S4_>
  406bfe:	c9                   	leaveq 
  406bff:	c3                   	retq   

0000000000406c00 <_ZNSt10_Iter_baseIPN2cv3MatELb0EE7_S_baseES2_>:
  406c00:	55                   	push   %rbp
  406c01:	48 89 e5             	mov    %rsp,%rbp
  406c04:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406c08:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406c0c:	5d                   	pop    %rbp
  406c0d:	c3                   	retq   

0000000000406c0e <_ZSt12__niter_baseIPN2cv3MatEENSt11_Niter_baseIT_E13iterator_typeES4_>:
  406c0e:	55                   	push   %rbp
  406c0f:	48 89 e5             	mov    %rsp,%rbp
  406c12:	48 83 ec 10          	sub    $0x10,%rsp
  406c16:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406c1a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406c1e:	48 89 c7             	mov    %rax,%rdi
  406c21:	e8 da ff ff ff       	callq  406c00 <_ZNSt10_Iter_baseIPN2cv3MatELb0EE7_S_baseES2_>
  406c26:	c9                   	leaveq 
  406c27:	c3                   	retq   

0000000000406c28 <_ZSt22__copy_move_backward_aILb0EPN2cv3MatES2_ET1_T0_S4_S3_>:
  406c28:	55                   	push   %rbp
  406c29:	48 89 e5             	mov    %rsp,%rbp
  406c2c:	48 83 ec 30          	sub    $0x30,%rsp
  406c30:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  406c34:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  406c38:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
  406c3c:	c6 45 ff 00          	movb   $0x0,-0x1(%rbp)
  406c40:	48 8b 55 d8          	mov    -0x28(%rbp),%rdx
  406c44:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  406c48:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  406c4c:	48 89 ce             	mov    %rcx,%rsi
  406c4f:	48 89 c7             	mov    %rax,%rdi
  406c52:	e8 4f 01 00 00       	callq  406da6 <_ZNSt20__copy_move_backwardILb0ELb0ESt26random_access_iterator_tagE13__copy_move_bIPN2cv3MatES5_EET0_T_S7_S6_>
  406c57:	c9                   	leaveq 
  406c58:	c3                   	retq   

0000000000406c59 <_ZN9__gnu_cxx14__alloc_traitsISaIN2cv3MatEEE8max_sizeERKS3_>:
  406c59:	55                   	push   %rbp
  406c5a:	48 89 e5             	mov    %rsp,%rbp
  406c5d:	48 83 ec 10          	sub    $0x10,%rsp
  406c61:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406c65:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406c69:	48 89 c7             	mov    %rax,%rdi
  406c6c:	e8 11 00 00 00       	callq  406c82 <_ZNK9__gnu_cxx13new_allocatorIN2cv3MatEE8max_sizeEv>
  406c71:	c9                   	leaveq 
  406c72:	c3                   	retq   
  406c73:	90                   	nop

0000000000406c74 <_ZNKSt12_Vector_baseIN2cv3MatESaIS1_EE19_M_get_Tp_allocatorEv>:
  406c74:	55                   	push   %rbp
  406c75:	48 89 e5             	mov    %rsp,%rbp
  406c78:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406c7c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406c80:	5d                   	pop    %rbp
  406c81:	c3                   	retq   

0000000000406c82 <_ZNK9__gnu_cxx13new_allocatorIN2cv3MatEE8max_sizeEv>:
  406c82:	55                   	push   %rbp
  406c83:	48 89 e5             	mov    %rsp,%rbp
  406c86:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406c8a:	48 b8 aa aa aa aa aa 	movabs $0x2aaaaaaaaaaaaaa,%rax
  406c91:	aa aa 02 
  406c94:	5d                   	pop    %rbp
  406c95:	c3                   	retq   

0000000000406c96 <_ZSt18uninitialized_copyIPN2cv3MatES2_ET0_T_S4_S3_>:
  406c96:	55                   	push   %rbp
  406c97:	48 89 e5             	mov    %rsp,%rbp
  406c9a:	48 83 ec 20          	sub    $0x20,%rsp
  406c9e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406ca2:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  406ca6:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  406caa:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  406cae:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  406cb2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406cb6:	48 89 ce             	mov    %rcx,%rsi
  406cb9:	48 89 c7             	mov    %rax,%rdi
  406cbc:	e8 51 01 00 00       	callq  406e12 <_ZNSt20__uninitialized_copyILb0EE13__uninit_copyIPN2cv3MatES4_EET0_T_S6_S5_>
  406cc1:	c9                   	leaveq 
  406cc2:	c3                   	retq   
  406cc3:	90                   	nop

0000000000406cc4 <_ZN9__gnu_cxx13new_allocatorIfE8allocateEmPKv>:
  406cc4:	55                   	push   %rbp
  406cc5:	48 89 e5             	mov    %rsp,%rbp
  406cc8:	48 83 ec 20          	sub    $0x20,%rsp
  406ccc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406cd0:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  406cd4:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  406cd8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406cdc:	48 89 c7             	mov    %rax,%rdi
  406cdf:	e8 bc 01 00 00       	callq  406ea0 <_ZNK9__gnu_cxx13new_allocatorIfE8max_sizeEv>
  406ce4:	48 3b 45 f0          	cmp    -0x10(%rbp),%rax
  406ce8:	0f 92 c0             	setb   %al
  406ceb:	84 c0                	test   %al,%al
  406ced:	74 05                	je     406cf4 <_ZN9__gnu_cxx13new_allocatorIfE8allocateEmPKv+0x30>
  406cef:	e8 4c aa ff ff       	callq  401740 <_ZSt17__throw_bad_allocv@plt>
  406cf4:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  406cf8:	48 c1 e0 02          	shl    $0x2,%rax
  406cfc:	48 89 c7             	mov    %rax,%rdi
  406cff:	e8 5c aa ff ff       	callq  401760 <_Znwm@plt>
  406d04:	c9                   	leaveq 
  406d05:	c3                   	retq   

0000000000406d06 <_ZNSt10_Iter_baseIPhLb0EE7_S_baseES0_>:
  406d06:	55                   	push   %rbp
  406d07:	48 89 e5             	mov    %rsp,%rbp
  406d0a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406d0e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406d12:	5d                   	pop    %rbp
  406d13:	c3                   	retq   

0000000000406d14 <_ZSt12__niter_baseIPhENSt11_Niter_baseIT_E13iterator_typeES2_>:
  406d14:	55                   	push   %rbp
  406d15:	48 89 e5             	mov    %rsp,%rbp
  406d18:	48 83 ec 10          	sub    $0x10,%rsp
  406d1c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406d20:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406d24:	48 89 c7             	mov    %rax,%rdi
  406d27:	e8 da ff ff ff       	callq  406d06 <_ZNSt10_Iter_baseIPhLb0EE7_S_baseES0_>
  406d2c:	c9                   	leaveq 
  406d2d:	c3                   	retq   

0000000000406d2e <_ZSt12__niter_baseIPfENSt11_Niter_baseIT_E13iterator_typeES2_>:
  406d2e:	55                   	push   %rbp
  406d2f:	48 89 e5             	mov    %rsp,%rbp
  406d32:	48 83 ec 10          	sub    $0x10,%rsp
  406d36:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406d3a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406d3e:	48 89 c7             	mov    %rax,%rdi
  406d41:	e8 6e 01 00 00       	callq  406eb4 <_ZNSt10_Iter_baseIPfLb0EE7_S_baseES0_>
  406d46:	c9                   	leaveq 
  406d47:	c3                   	retq   

0000000000406d48 <_ZSt13__copy_move_aILb0EPhPfET1_T0_S3_S2_>:
  406d48:	55                   	push   %rbp
  406d49:	48 89 e5             	mov    %rsp,%rbp
  406d4c:	48 83 ec 30          	sub    $0x30,%rsp
  406d50:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  406d54:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  406d58:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
  406d5c:	c6 45 ff 00          	movb   $0x0,-0x1(%rbp)
  406d60:	48 8b 55 d8          	mov    -0x28(%rbp),%rdx
  406d64:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  406d68:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  406d6c:	48 89 ce             	mov    %rcx,%rsi
  406d6f:	48 89 c7             	mov    %rax,%rdi
  406d72:	e8 4b 01 00 00       	callq  406ec2 <_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPhPfEET0_T_S6_S5_>
  406d77:	c9                   	leaveq 
  406d78:	c3                   	retq   

0000000000406d79 <_ZNSt20__uninitialized_copyILb1EE13__uninit_copyIPhPfEET0_T_S5_S4_>:
  406d79:	55                   	push   %rbp
  406d7a:	48 89 e5             	mov    %rsp,%rbp
  406d7d:	48 83 ec 20          	sub    $0x20,%rsp
  406d81:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406d85:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  406d89:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  406d8d:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  406d91:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  406d95:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406d99:	48 89 ce             	mov    %rcx,%rsi
  406d9c:	48 89 c7             	mov    %rax,%rdi
  406d9f:	e8 aa fa ff ff       	callq  40684e <_ZSt4copyIPhPfET0_T_S3_S2_>
  406da4:	c9                   	leaveq 
  406da5:	c3                   	retq   

0000000000406da6 <_ZNSt20__copy_move_backwardILb0ELb0ESt26random_access_iterator_tagE13__copy_move_bIPN2cv3MatES5_EET0_T_S7_S6_>:
  406da6:	55                   	push   %rbp
  406da7:	48 89 e5             	mov    %rsp,%rbp
  406daa:	48 83 ec 30          	sub    $0x30,%rsp
  406dae:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  406db2:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  406db6:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
  406dba:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
  406dbe:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  406dc2:	48 29 c2             	sub    %rax,%rdx
  406dc5:	48 89 d0             	mov    %rdx,%rax
  406dc8:	48 c1 f8 05          	sar    $0x5,%rax
  406dcc:	48 89 c2             	mov    %rax,%rdx
  406dcf:	48 b8 ab aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaab,%rax
  406dd6:	aa aa aa 
  406dd9:	48 0f af c2          	imul   %rdx,%rax
  406ddd:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  406de1:	eb 22                	jmp    406e05 <_ZNSt20__copy_move_backwardILb0ELb0ESt26random_access_iterator_tagE13__copy_move_bIPN2cv3MatES5_EET0_T_S7_S6_+0x5f>
  406de3:	48 83 6d e0 60       	subq   $0x60,-0x20(%rbp)
  406de8:	48 83 6d d8 60       	subq   $0x60,-0x28(%rbp)
  406ded:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
  406df1:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  406df5:	48 89 d6             	mov    %rdx,%rsi
  406df8:	48 89 c7             	mov    %rax,%rdi
  406dfb:	e8 fc e4 ff ff       	callq  4052fc <_ZN2cv3MataSERKS0_>
  406e00:	48 83 6d f8 01       	subq   $0x1,-0x8(%rbp)
  406e05:	48 83 7d f8 00       	cmpq   $0x0,-0x8(%rbp)
  406e0a:	7f d7                	jg     406de3 <_ZNSt20__copy_move_backwardILb0ELb0ESt26random_access_iterator_tagE13__copy_move_bIPN2cv3MatES5_EET0_T_S7_S6_+0x3d>
  406e0c:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  406e10:	c9                   	leaveq 
  406e11:	c3                   	retq   

0000000000406e12 <_ZNSt20__uninitialized_copyILb0EE13__uninit_copyIPN2cv3MatES4_EET0_T_S6_S5_>:
  406e12:	55                   	push   %rbp
  406e13:	48 89 e5             	mov    %rsp,%rbp
  406e16:	53                   	push   %rbx
  406e17:	48 83 ec 38          	sub    $0x38,%rsp
  406e1b:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  406e1f:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  406e23:	48 89 55 c8          	mov    %rdx,-0x38(%rbp)
  406e27:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  406e2b:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  406e2f:	eb 25                	jmp    406e56 <_ZNSt20__uninitialized_copyILb0EE13__uninit_copyIPN2cv3MatES4_EET0_T_S6_S5_+0x44>
  406e31:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  406e35:	48 89 c7             	mov    %rax,%rdi
  406e38:	e8 71 fc ff ff       	callq  406aae <_ZSt11__addressofIN2cv3MatEEPT_RS2_>
  406e3d:	48 8b 55 d8          	mov    -0x28(%rbp),%rdx
  406e41:	48 89 d6             	mov    %rdx,%rsi
  406e44:	48 89 c7             	mov    %rax,%rdi
  406e47:	e8 cc 00 00 00       	callq  406f18 <_ZSt10_ConstructIN2cv3MatES1_EvPT_RKT0_>
  406e4c:	48 83 45 d8 60       	addq   $0x60,-0x28(%rbp)
  406e51:	48 83 45 e8 60       	addq   $0x60,-0x18(%rbp)
  406e56:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  406e5a:	48 3b 45 d0          	cmp    -0x30(%rbp),%rax
  406e5e:	75 d1                	jne    406e31 <_ZNSt20__uninitialized_copyILb0EE13__uninit_copyIPN2cv3MatES4_EET0_T_S6_S5_+0x1f>
  406e60:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  406e64:	eb 33                	jmp    406e99 <_ZNSt20__uninitialized_copyILb0EE13__uninit_copyIPN2cv3MatES4_EET0_T_S6_S5_+0x87>
  406e66:	48 89 c7             	mov    %rax,%rdi
  406e69:	e8 92 a8 ff ff       	callq  401700 <__cxa_begin_catch@plt>
  406e6e:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  406e72:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  406e76:	48 89 d6             	mov    %rdx,%rsi
  406e79:	48 89 c7             	mov    %rax,%rdi
  406e7c:	e8 fd f1 ff ff       	callq  40607e <_ZSt8_DestroyIPN2cv3MatEEvT_S3_>
  406e81:	e8 2a aa ff ff       	callq  4018b0 <__cxa_rethrow@plt>
  406e86:	48 89 c3             	mov    %rax,%rbx
  406e89:	e8 02 aa ff ff       	callq  401890 <__cxa_end_catch@plt>
  406e8e:	48 89 d8             	mov    %rbx,%rax
  406e91:	48 89 c7             	mov    %rax,%rdi
  406e94:	e8 a7 a9 ff ff       	callq  401840 <_Unwind_Resume@plt>
  406e99:	48 83 c4 38          	add    $0x38,%rsp
  406e9d:	5b                   	pop    %rbx
  406e9e:	5d                   	pop    %rbp
  406e9f:	c3                   	retq   

0000000000406ea0 <_ZNK9__gnu_cxx13new_allocatorIfE8max_sizeEv>:
  406ea0:	55                   	push   %rbp
  406ea1:	48 89 e5             	mov    %rsp,%rbp
  406ea4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406ea8:	48 b8 ff ff ff ff ff 	movabs $0x3fffffffffffffff,%rax
  406eaf:	ff ff 3f 
  406eb2:	5d                   	pop    %rbp
  406eb3:	c3                   	retq   

0000000000406eb4 <_ZNSt10_Iter_baseIPfLb0EE7_S_baseES0_>:
  406eb4:	55                   	push   %rbp
  406eb5:	48 89 e5             	mov    %rsp,%rbp
  406eb8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  406ebc:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  406ec0:	5d                   	pop    %rbp
  406ec1:	c3                   	retq   

0000000000406ec2 <_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPhPfEET0_T_S6_S5_>:
  406ec2:	55                   	push   %rbp
  406ec3:	48 89 e5             	mov    %rsp,%rbp
  406ec6:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  406eca:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  406ece:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
  406ed2:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
  406ed6:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  406eda:	48 29 c2             	sub    %rax,%rdx
  406edd:	48 89 d0             	mov    %rdx,%rax
  406ee0:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  406ee4:	eb 25                	jmp    406f0b <_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPhPfEET0_T_S6_S5_+0x49>
  406ee6:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  406eea:	0f b6 00             	movzbl (%rax),%eax
  406eed:	0f b6 c0             	movzbl %al,%eax
  406ef0:	c5 fa 2a c0          	vcvtsi2ss %eax,%xmm0,%xmm0
  406ef4:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  406ef8:	c5 fa 11 00          	vmovss %xmm0,(%rax)
  406efc:	48 83 45 e8 01       	addq   $0x1,-0x18(%rbp)
  406f01:	48 83 45 d8 04       	addq   $0x4,-0x28(%rbp)
  406f06:	48 83 6d f8 01       	subq   $0x1,-0x8(%rbp)
  406f0b:	48 83 7d f8 00       	cmpq   $0x0,-0x8(%rbp)
  406f10:	7f d4                	jg     406ee6 <_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPhPfEET0_T_S6_S5_+0x24>
  406f12:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  406f16:	5d                   	pop    %rbp
  406f17:	c3                   	retq   

0000000000406f18 <_ZSt10_ConstructIN2cv3MatES1_EvPT_RKT0_>:
  406f18:	55                   	push   %rbp
  406f19:	48 89 e5             	mov    %rsp,%rbp
  406f1c:	41 55                	push   %r13
  406f1e:	41 54                	push   %r12
  406f20:	53                   	push   %rbx
  406f21:	48 83 ec 18          	sub    $0x18,%rsp
  406f25:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  406f29:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  406f2d:	4c 8b 65 d8          	mov    -0x28(%rbp),%r12
  406f31:	4c 89 e6             	mov    %r12,%rsi
  406f34:	bf 60 00 00 00       	mov    $0x60,%edi
  406f39:	e8 f0 de ff ff       	callq  404e2e <_ZnwmPv>
  406f3e:	48 89 c3             	mov    %rax,%rbx
  406f41:	48 85 db             	test   %rbx,%rbx
  406f44:	74 11                	je     406f57 <_ZSt10_ConstructIN2cv3MatES1_EvPT_RKT0_+0x3f>
  406f46:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  406f4a:	48 89 c6             	mov    %rax,%rsi
  406f4d:	48 89 df             	mov    %rbx,%rdi
  406f50:	e8 f3 df ff ff       	callq  404f48 <_ZN2cv3MatC1ERKS0_>
  406f55:	eb 1b                	jmp    406f72 <_ZSt10_ConstructIN2cv3MatES1_EvPT_RKT0_+0x5a>
  406f57:	eb 19                	jmp    406f72 <_ZSt10_ConstructIN2cv3MatES1_EvPT_RKT0_+0x5a>
  406f59:	49 89 c5             	mov    %rax,%r13
  406f5c:	4c 89 e6             	mov    %r12,%rsi
  406f5f:	48 89 df             	mov    %rbx,%rdi
  406f62:	e8 d9 de ff ff       	callq  404e40 <_ZdlPvS_>
  406f67:	4c 89 e8             	mov    %r13,%rax
  406f6a:	48 89 c7             	mov    %rax,%rdi
  406f6d:	e8 ce a8 ff ff       	callq  401840 <_Unwind_Resume@plt>
  406f72:	48 83 c4 18          	add    $0x18,%rsp
  406f76:	5b                   	pop    %rbx
  406f77:	41 5c                	pop    %r12
  406f79:	41 5d                	pop    %r13
  406f7b:	5d                   	pop    %rbp
  406f7c:	c3                   	retq   
  406f7d:	0f 1f 00             	nopl   (%rax)

0000000000406f80 <__libc_csu_init>:
  406f80:	41 57                	push   %r15
  406f82:	41 89 ff             	mov    %edi,%r15d
  406f85:	41 56                	push   %r14
  406f87:	49 89 f6             	mov    %rsi,%r14
  406f8a:	41 55                	push   %r13
  406f8c:	49 89 d5             	mov    %rdx,%r13
  406f8f:	41 54                	push   %r12
  406f91:	4c 8d 25 40 1d 20 00 	lea    0x201d40(%rip),%r12        # 608cd8 <__frame_dummy_init_array_entry>
  406f98:	55                   	push   %rbp
  406f99:	48 8d 2d 48 1d 20 00 	lea    0x201d48(%rip),%rbp        # 608ce8 <__init_array_end>
  406fa0:	53                   	push   %rbx
  406fa1:	4c 29 e5             	sub    %r12,%rbp
  406fa4:	31 db                	xor    %ebx,%ebx
  406fa6:	48 c1 fd 03          	sar    $0x3,%rbp
  406faa:	48 83 ec 08          	sub    $0x8,%rsp
  406fae:	e8 1d a7 ff ff       	callq  4016d0 <_init>
  406fb3:	48 85 ed             	test   %rbp,%rbp
  406fb6:	74 1e                	je     406fd6 <__libc_csu_init+0x56>
  406fb8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  406fbf:	00 
  406fc0:	4c 89 ea             	mov    %r13,%rdx
  406fc3:	4c 89 f6             	mov    %r14,%rsi
  406fc6:	44 89 ff             	mov    %r15d,%edi
  406fc9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  406fcd:	48 83 c3 01          	add    $0x1,%rbx
  406fd1:	48 39 eb             	cmp    %rbp,%rbx
  406fd4:	75 ea                	jne    406fc0 <__libc_csu_init+0x40>
  406fd6:	48 83 c4 08          	add    $0x8,%rsp
  406fda:	5b                   	pop    %rbx
  406fdb:	5d                   	pop    %rbp
  406fdc:	41 5c                	pop    %r12
  406fde:	41 5d                	pop    %r13
  406fe0:	41 5e                	pop    %r14
  406fe2:	41 5f                	pop    %r15
  406fe4:	c3                   	retq   
  406fe5:	90                   	nop
  406fe6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  406fed:	00 00 00 

0000000000406ff0 <__libc_csu_fini>:
  406ff0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000406ff4 <_fini>:
  406ff4:	48 83 ec 08          	sub    $0x8,%rsp
  406ff8:	48 83 c4 08          	add    $0x8,%rsp
  406ffc:	c3                   	retq   
