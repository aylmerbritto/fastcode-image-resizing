
./build/parallelNN:     file format elf64-x86-64


Disassembly of section .init:

00000000004014a8 <_init>:
  4014a8:	48 83 ec 08          	sub    $0x8,%rsp
  4014ac:	48 8b 05 45 3b 20 00 	mov    0x203b45(%rip),%rax        # 604ff8 <__gmon_start__>
  4014b3:	48 85 c0             	test   %rax,%rax
  4014b6:	74 05                	je     4014bd <_init+0x15>
  4014b8:	e8 b3 00 00 00       	callq  401570 <__gmon_start__@plt>
  4014bd:	48 83 c4 08          	add    $0x8,%rsp
  4014c1:	c3                   	retq   

Disassembly of section .plt:

00000000004014d0 <.plt>:
  4014d0:	ff 35 32 3b 20 00    	pushq  0x203b32(%rip)        # 605008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4014d6:	ff 25 34 3b 20 00    	jmpq   *0x203b34(%rip)        # 605010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4014dc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004014e0 <__cxa_begin_catch@plt>:
  4014e0:	ff 25 32 3b 20 00    	jmpq   *0x203b32(%rip)        # 605018 <__cxa_begin_catch@CXXABI_1.3>
  4014e6:	68 00 00 00 00       	pushq  $0x0
  4014eb:	e9 e0 ff ff ff       	jmpq   4014d0 <.plt>

00000000004014f0 <memcpy@plt>:
  4014f0:	ff 25 2a 3b 20 00    	jmpq   *0x203b2a(%rip)        # 605020 <memcpy@GLIBC_2.14>
  4014f6:	68 01 00 00 00       	pushq  $0x1
  4014fb:	e9 d0 ff ff ff       	jmpq   4014d0 <.plt>

0000000000401500 <_ZN2cv3Mat8copySizeERKS0_@plt>:
  401500:	ff 25 22 3b 20 00    	jmpq   *0x203b22(%rip)        # 605028 <_ZN2cv3Mat8copySizeERKS0_>
  401506:	68 02 00 00 00       	pushq  $0x2
  40150b:	e9 c0 ff ff ff       	jmpq   4014d0 <.plt>

0000000000401510 <_ZSt17__throw_bad_allocv@plt>:
  401510:	ff 25 1a 3b 20 00    	jmpq   *0x203b1a(%rip)        # 605030 <_ZSt17__throw_bad_allocv@GLIBCXX_3.4>
  401516:	68 03 00 00 00       	pushq  $0x3
  40151b:	e9 b0 ff ff ff       	jmpq   4014d0 <.plt>

0000000000401520 <_Znwm@plt>:
  401520:	ff 25 12 3b 20 00    	jmpq   *0x203b12(%rip)        # 605038 <_Znwm@GLIBCXX_3.4>
  401526:	68 04 00 00 00       	pushq  $0x4
  40152b:	e9 a0 ff ff ff       	jmpq   4014d0 <.plt>

0000000000401530 <free@plt>:
  401530:	ff 25 0a 3b 20 00    	jmpq   *0x203b0a(%rip)        # 605040 <free@GLIBC_2.2.5>
  401536:	68 05 00 00 00       	pushq  $0x5
  40153b:	e9 90 ff ff ff       	jmpq   4014d0 <.plt>

0000000000401540 <_ZN2cv12_OutputArrayC1ERNS_3MatE@plt>:
  401540:	ff 25 02 3b 20 00    	jmpq   *0x203b02(%rip)        # 605048 <_ZN2cv12_OutputArrayC1ERNS_3MatE>
  401546:	68 06 00 00 00       	pushq  $0x6
  40154b:	e9 80 ff ff ff       	jmpq   4014d0 <.plt>

0000000000401550 <_ZNSolsEi@plt>:
  401550:	ff 25 fa 3a 20 00    	jmpq   *0x203afa(%rip)        # 605050 <_ZNSolsEi@GLIBCXX_3.4>
  401556:	68 07 00 00 00       	pushq  $0x7
  40155b:	e9 70 ff ff ff       	jmpq   4014d0 <.plt>

0000000000401560 <_ZSt16__throw_bad_castv@plt>:
  401560:	ff 25 f2 3a 20 00    	jmpq   *0x203af2(%rip)        # 605058 <_ZSt16__throw_bad_castv@GLIBCXX_3.4>
  401566:	68 08 00 00 00       	pushq  $0x8
  40156b:	e9 60 ff ff ff       	jmpq   4014d0 <.plt>

0000000000401570 <__gmon_start__@plt>:
  401570:	ff 25 ea 3a 20 00    	jmpq   *0x203aea(%rip)        # 605060 <__gmon_start__>
  401576:	68 09 00 00 00       	pushq  $0x9
  40157b:	e9 50 ff ff ff       	jmpq   4014d0 <.plt>

0000000000401580 <omp_get_num_threads@plt>:
  401580:	ff 25 e2 3a 20 00    	jmpq   *0x203ae2(%rip)        # 605068 <omp_get_num_threads@OMP_1.0>
  401586:	68 0a 00 00 00       	pushq  $0xa
  40158b:	e9 40 ff ff ff       	jmpq   4014d0 <.plt>

0000000000401590 <omp_get_thread_num@plt>:
  401590:	ff 25 da 3a 20 00    	jmpq   *0x203ada(%rip)        # 605070 <omp_get_thread_num@OMP_1.0>
  401596:	68 0b 00 00 00       	pushq  $0xb
  40159b:	e9 30 ff ff ff       	jmpq   4014d0 <.plt>

00000000004015a0 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>:
  4015a0:	ff 25 d2 3a 20 00    	jmpq   *0x203ad2(%rip)        # 605078 <_ZNKSt5ctypeIcE13_M_widen_initEv@GLIBCXX_3.4.11>
  4015a6:	68 0c 00 00 00       	pushq  $0xc
  4015ab:	e9 20 ff ff ff       	jmpq   4014d0 <.plt>

00000000004015b0 <_ZNSo5flushEv@plt>:
  4015b0:	ff 25 ca 3a 20 00    	jmpq   *0x203aca(%rip)        # 605080 <_ZNSo5flushEv@GLIBCXX_3.4>
  4015b6:	68 0d 00 00 00       	pushq  $0xd
  4015bb:	e9 10 ff ff ff       	jmpq   4014d0 <.plt>

00000000004015c0 <_ZN2cv3Mat10deallocateEv@plt>:
  4015c0:	ff 25 c2 3a 20 00    	jmpq   *0x203ac2(%rip)        # 605088 <_ZN2cv3Mat10deallocateEv>
  4015c6:	68 0e 00 00 00       	pushq  $0xe
  4015cb:	e9 00 ff ff ff       	jmpq   4014d0 <.plt>

00000000004015d0 <_ZdlPv@plt>:
  4015d0:	ff 25 ba 3a 20 00    	jmpq   *0x203aba(%rip)        # 605090 <_ZdlPv@GLIBCXX_3.4>
  4015d6:	68 0f 00 00 00       	pushq  $0xf
  4015db:	e9 f0 fe ff ff       	jmpq   4014d0 <.plt>

00000000004015e0 <_ZNSo3putEc@plt>:
  4015e0:	ff 25 b2 3a 20 00    	jmpq   *0x203ab2(%rip)        # 605098 <_ZNSo3putEc@GLIBCXX_3.4>
  4015e6:	68 10 00 00 00       	pushq  $0x10
  4015eb:	e9 e0 fe ff ff       	jmpq   4014d0 <.plt>

00000000004015f0 <__gxx_personality_v0@plt>:
  4015f0:	ff 25 aa 3a 20 00    	jmpq   *0x203aaa(%rip)        # 6050a0 <__gxx_personality_v0@CXXABI_1.3>
  4015f6:	68 11 00 00 00       	pushq  $0x11
  4015fb:	e9 d0 fe ff ff       	jmpq   4014d0 <.plt>

0000000000401600 <_ZN2cv7imwriteERKSsRKNS_11_InputArrayERKSt6vectorIiSaIiEE@plt>:
  401600:	ff 25 a2 3a 20 00    	jmpq   *0x203aa2(%rip)        # 6050a8 <_ZN2cv7imwriteERKSsRKNS_11_InputArrayERKSt6vectorIiSaIiEE>
  401606:	68 12 00 00 00       	pushq  $0x12
  40160b:	e9 c0 fe ff ff       	jmpq   4014d0 <.plt>

0000000000401610 <GOMP_parallel_end@plt>:
  401610:	ff 25 9a 3a 20 00    	jmpq   *0x203a9a(%rip)        # 6050b0 <GOMP_parallel_end@GOMP_1.0>
  401616:	68 13 00 00 00       	pushq  $0x13
  40161b:	e9 b0 fe ff ff       	jmpq   4014d0 <.plt>

0000000000401620 <_Unwind_Resume@plt>:
  401620:	ff 25 92 3a 20 00    	jmpq   *0x203a92(%rip)        # 6050b8 <_Unwind_Resume@GCC_3.0>
  401626:	68 14 00 00 00       	pushq  $0x14
  40162b:	e9 a0 fe ff ff       	jmpq   4014d0 <.plt>

0000000000401630 <_ZNSt8ios_base4InitD1Ev@plt>:
  401630:	ff 25 8a 3a 20 00    	jmpq   *0x203a8a(%rip)        # 6050c0 <_ZNSt8ios_base4InitD1Ev@GLIBCXX_3.4>
  401636:	68 15 00 00 00       	pushq  $0x15
  40163b:	e9 90 fe ff ff       	jmpq   4014d0 <.plt>

0000000000401640 <_ZN2cv11_InputArrayC1ERKNS_3MatE@plt>:
  401640:	ff 25 82 3a 20 00    	jmpq   *0x203a82(%rip)        # 6050c8 <_ZN2cv11_InputArrayC1ERKNS_3MatE>
  401646:	68 16 00 00 00       	pushq  $0x16
  40164b:	e9 80 fe ff ff       	jmpq   4014d0 <.plt>

0000000000401650 <_ZN2cv5mergeERKSt6vectorINS_3MatESaIS1_EERKNS_12_OutputArrayE@plt>:
  401650:	ff 25 7a 3a 20 00    	jmpq   *0x203a7a(%rip)        # 6050d0 <_ZN2cv5mergeERKSt6vectorINS_3MatESaIS1_EERKNS_12_OutputArrayE>
  401656:	68 17 00 00 00       	pushq  $0x17
  40165b:	e9 70 fe ff ff       	jmpq   4014d0 <.plt>

0000000000401660 <_ZNSo9_M_insertIdEERSoT_@plt>:
  401660:	ff 25 72 3a 20 00    	jmpq   *0x203a72(%rip)        # 6050d8 <_ZNSo9_M_insertIdEERSoT_@GLIBCXX_3.4.9>
  401666:	68 18 00 00 00       	pushq  $0x18
  40166b:	e9 60 fe ff ff       	jmpq   4014d0 <.plt>

0000000000401670 <__cxa_end_catch@plt>:
  401670:	ff 25 6a 3a 20 00    	jmpq   *0x203a6a(%rip)        # 6050e0 <__cxa_end_catch@CXXABI_1.3>
  401676:	68 19 00 00 00       	pushq  $0x19
  40167b:	e9 50 fe ff ff       	jmpq   4014d0 <.plt>

0000000000401680 <_ZNSs4_Rep10_M_disposeERKSaIcE@plt>:
  401680:	ff 25 62 3a 20 00    	jmpq   *0x203a62(%rip)        # 6050e8 <_ZNSs4_Rep10_M_disposeERKSaIcE@GLIBCXX_3.4>
  401686:	68 1a 00 00 00       	pushq  $0x1a
  40168b:	e9 40 fe ff ff       	jmpq   4014d0 <.plt>

0000000000401690 <__cxa_rethrow@plt>:
  401690:	ff 25 5a 3a 20 00    	jmpq   *0x203a5a(%rip)        # 6050f0 <__cxa_rethrow@CXXABI_1.3>
  401696:	68 1b 00 00 00       	pushq  $0x1b
  40169b:	e9 30 fe ff ff       	jmpq   4014d0 <.plt>

00000000004016a0 <calloc@plt>:
  4016a0:	ff 25 52 3a 20 00    	jmpq   *0x203a52(%rip)        # 6050f8 <calloc@GLIBC_2.2.5>
  4016a6:	68 1c 00 00 00       	pushq  $0x1c
  4016ab:	e9 20 fe ff ff       	jmpq   4014d0 <.plt>

00000000004016b0 <__cxa_atexit@plt>:
  4016b0:	ff 25 4a 3a 20 00    	jmpq   *0x203a4a(%rip)        # 605100 <__cxa_atexit@GLIBC_2.2.5>
  4016b6:	68 1d 00 00 00       	pushq  $0x1d
  4016bb:	e9 10 fe ff ff       	jmpq   4014d0 <.plt>

00000000004016c0 <omp_set_num_threads@plt>:
  4016c0:	ff 25 42 3a 20 00    	jmpq   *0x203a42(%rip)        # 605108 <omp_set_num_threads@OMP_1.0>
  4016c6:	68 1e 00 00 00       	pushq  $0x1e
  4016cb:	e9 00 fe ff ff       	jmpq   4014d0 <.plt>

00000000004016d0 <_ZN2cv8fastFreeEPv@plt>:
  4016d0:	ff 25 3a 3a 20 00    	jmpq   *0x203a3a(%rip)        # 605110 <_ZN2cv8fastFreeEPv>
  4016d6:	68 1f 00 00 00       	pushq  $0x1f
  4016db:	e9 f0 fd ff ff       	jmpq   4014d0 <.plt>

00000000004016e0 <__pthread_key_create@plt>:
  4016e0:	ff 25 32 3a 20 00    	jmpq   *0x203a32(%rip)        # 605118 <__pthread_key_create@GLIBC_2.2.5>
  4016e6:	68 20 00 00 00       	pushq  $0x20
  4016eb:	e9 e0 fd ff ff       	jmpq   4014d0 <.plt>

00000000004016f0 <__libc_start_main@plt>:
  4016f0:	ff 25 2a 3a 20 00    	jmpq   *0x203a2a(%rip)        # 605120 <__libc_start_main@GLIBC_2.2.5>
  4016f6:	68 21 00 00 00       	pushq  $0x21
  4016fb:	e9 d0 fd ff ff       	jmpq   4014d0 <.plt>

0000000000401700 <stpcpy@plt>:
  401700:	ff 25 22 3a 20 00    	jmpq   *0x203a22(%rip)        # 605128 <stpcpy@GLIBC_2.2.5>
  401706:	68 22 00 00 00       	pushq  $0x22
  40170b:	e9 c0 fd ff ff       	jmpq   4014d0 <.plt>

0000000000401710 <_ZNSs4_Rep10_M_destroyERKSaIcE@plt>:
  401710:	ff 25 1a 3a 20 00    	jmpq   *0x203a1a(%rip)        # 605130 <_ZNSs4_Rep10_M_destroyERKSaIcE@GLIBCXX_3.4>
  401716:	68 23 00 00 00       	pushq  $0x23
  40171b:	e9 b0 fd ff ff       	jmpq   4014d0 <.plt>

0000000000401720 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>:
  401720:	ff 25 12 3a 20 00    	jmpq   *0x203a12(%rip)        # 605138 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@GLIBCXX_3.4.9>
  401726:	68 24 00 00 00       	pushq  $0x24
  40172b:	e9 a0 fd ff ff       	jmpq   4014d0 <.plt>

0000000000401730 <_ZNSt8ios_base4InitC1Ev@plt>:
  401730:	ff 25 0a 3a 20 00    	jmpq   *0x203a0a(%rip)        # 605140 <_ZNSt8ios_base4InitC1Ev@GLIBCXX_3.4>
  401736:	68 25 00 00 00       	pushq  $0x25
  40173b:	e9 90 fd ff ff       	jmpq   4014d0 <.plt>

0000000000401740 <_ZN2cv5splitERKNS_3MatEPS0_@plt>:
  401740:	ff 25 02 3a 20 00    	jmpq   *0x203a02(%rip)        # 605148 <_ZN2cv5splitERKNS_3MatEPS0_>
  401746:	68 26 00 00 00       	pushq  $0x26
  40174b:	e9 80 fd ff ff       	jmpq   4014d0 <.plt>

0000000000401750 <_ZN2cv6imreadERKSsi@plt>:
  401750:	ff 25 fa 39 20 00    	jmpq   *0x2039fa(%rip)        # 605150 <_ZN2cv6imreadERKSsi>
  401756:	68 27 00 00 00       	pushq  $0x27
  40175b:	e9 70 fd ff ff       	jmpq   4014d0 <.plt>

0000000000401760 <_ZNSsC1EPKcRKSaIcE@plt>:
  401760:	ff 25 f2 39 20 00    	jmpq   *0x2039f2(%rip)        # 605158 <_ZNSsC1EPKcRKSaIcE@GLIBCXX_3.4>
  401766:	68 28 00 00 00       	pushq  $0x28
  40176b:	e9 60 fd ff ff       	jmpq   4014d0 <.plt>

0000000000401770 <GOMP_parallel_start@plt>:
  401770:	ff 25 ea 39 20 00    	jmpq   *0x2039ea(%rip)        # 605160 <GOMP_parallel_start@GOMP_1.0>
  401776:	68 29 00 00 00       	pushq  $0x29
  40177b:	e9 50 fd ff ff       	jmpq   4014d0 <.plt>

Disassembly of section .text:

0000000000401780 <main>:
  401780:	55                   	push   %rbp
  401781:	be 04 00 00 00       	mov    $0x4,%esi
  401786:	bf 00 04 00 00       	mov    $0x400,%edi
  40178b:	48 89 e5             	mov    %rsp,%rbp
  40178e:	41 57                	push   %r15
  401790:	41 56                	push   %r14
  401792:	41 55                	push   %r13
  401794:	41 54                	push   %r12
  401796:	41 bc 20 00 00 00    	mov    $0x20,%r12d
  40179c:	53                   	push   %rbx
  40179d:	bb 20 00 00 00       	mov    $0x20,%ebx
  4017a2:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4017a6:	48 81 ec a0 01 00 00 	sub    $0x1a0,%rsp
  4017ad:	4c 89 5c 24 20       	mov    %r11,0x20(%rsp)
  4017b2:	4c 89 54 24 28       	mov    %r10,0x28(%rsp)
  4017b7:	e8 e4 fe ff ff       	callq  4016a0 <calloc@plt>
  4017bc:	be 04 00 00 00       	mov    $0x4,%esi
  4017c1:	bf 00 04 00 00       	mov    $0x400,%edi
  4017c6:	49 89 c7             	mov    %rax,%r15
  4017c9:	e8 d2 fe ff ff       	callq  4016a0 <calloc@plt>
  4017ce:	be 04 00 00 00       	mov    $0x4,%esi
  4017d3:	bf 00 04 00 00       	mov    $0x400,%edi
  4017d8:	49 89 c6             	mov    %rax,%r14
  4017db:	e8 c0 fe ff ff       	callq  4016a0 <calloc@plt>
  4017e0:	be 04 00 00 00       	mov    $0x4,%esi
  4017e5:	bf 00 01 00 00       	mov    $0x100,%edi
  4017ea:	49 89 c5             	mov    %rax,%r13
  4017ed:	e8 ae fe ff ff       	callq  4016a0 <calloc@plt>
  4017f2:	be 04 00 00 00       	mov    $0x4,%esi
  4017f7:	bf 00 01 00 00       	mov    $0x100,%edi
  4017fc:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
  401801:	e8 9a fe ff ff       	callq  4016a0 <calloc@plt>
  401806:	be 04 00 00 00       	mov    $0x4,%esi
  40180b:	bf 00 01 00 00       	mov    $0x100,%edi
  401810:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  401815:	e8 86 fe ff ff       	callq  4016a0 <calloc@plt>
  40181a:	48 8b 35 7f 39 20 00 	mov    0x20397f(%rip),%rsi        # 6051a0 <home>
  401821:	48 8d bc 24 c0 00 00 	lea    0xc0(%rsp),%rdi
  401828:	00 
  401829:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  40182e:	e8 cd fe ff ff       	callq  401700 <stpcpy@plt>
  401833:	4c 8b 4c 24 48       	mov    0x48(%rsp),%r9
  401838:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
  40183d:	48 ba 69 6e 70 75 74 	movabs $0x332f737475706e69,%rdx
  401844:	73 2f 33 
  401847:	4c 8b 44 24 40       	mov    0x40(%rsp),%r8
  40184c:	48 89 10             	mov    %rdx,(%rax)
  40184f:	48 ba 32 78 33 32 2e 	movabs $0x67706a2e32337832,%rdx
  401856:	6a 70 67 
  401859:	48 89 50 08          	mov    %rdx,0x8(%rax)
  40185d:	c6 40 10 00          	movb   $0x0,0x10(%rax)
  401861:	4c 89 ca             	mov    %r9,%rdx
  401864:	48 89 cf             	mov    %rcx,%rdi
  401867:	4c 89 4c 24 30       	mov    %r9,0x30(%rsp)
  40186c:	4c 89 c6             	mov    %r8,%rsi
  40186f:	48 89 4c 24 48       	mov    %rcx,0x48(%rsp)
  401874:	e8 f7 10 00 00       	callq  402970 <_Z11decodeImagePfS_S_>
  401879:	4c 8b 5c 24 20       	mov    0x20(%rsp),%r11
  40187e:	4c 8b 54 24 28       	mov    0x28(%rsp),%r10
  401883:	4c 8b 4c 24 30       	mov    0x30(%rsp),%r9
  401888:	4c 8b 44 24 40       	mov    0x40(%rsp),%r8
  40188d:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  401892:	c7 44 24 38 01 00 00 	movl   $0x1,0x38(%rsp)
  401899:	00 
  40189a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4018a0:	8b 7c 24 38          	mov    0x38(%rsp),%edi
  4018a4:	4c 89 4c 24 10       	mov    %r9,0x10(%rsp)
  4018a9:	4c 89 44 24 18       	mov    %r8,0x18(%rsp)
  4018ae:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
  4018b3:	4c 89 5c 24 28       	mov    %r11,0x28(%rsp)
  4018b8:	4c 89 54 24 30       	mov    %r10,0x30(%rsp)
  4018bd:	e8 fe fd ff ff       	callq  4016c0 <omp_set_num_threads@plt>
  4018c2:	c5 fb 10 1d a6 2b 00 	vmovsd 0x2ba6(%rip),%xmm3        # 404470 <__dso_handle+0x98>
  4018c9:	00 
  4018ca:	4c 8b 4c 24 10       	mov    0x10(%rsp),%r9
  4018cf:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  4018d4:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
  4018d9:	4c 8b 5c 24 28       	mov    0x28(%rsp),%r11
  4018de:	4c 8b 54 24 30       	mov    0x30(%rsp),%r10
  4018e3:	c5 fb 11 5c 24 48    	vmovsd %xmm3,0x48(%rsp)
  4018e9:	c7 44 24 40 64 00 00 	movl   $0x64,0x40(%rsp)
  4018f0:	00 
  4018f1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4018f8:	48 8d 74 24 60       	lea    0x60(%rsp),%rsi
  4018fd:	31 d2                	xor    %edx,%edx
  4018ff:	bf 10 1d 40 00       	mov    $0x401d10,%edi
  401904:	4c 89 54 24 60       	mov    %r10,0x60(%rsp)
  401909:	4c 89 5c 24 68       	mov    %r11,0x68(%rsp)
  40190e:	48 89 8c 24 90 00 00 	mov    %rcx,0x90(%rsp)
  401915:	00 
  401916:	4c 89 84 24 98 00 00 	mov    %r8,0x98(%rsp)
  40191d:	00 
  40191e:	4c 89 8c 24 a0 00 00 	mov    %r9,0xa0(%rsp)
  401925:	00 
  401926:	4c 89 7c 24 78       	mov    %r15,0x78(%rsp)
  40192b:	4c 89 b4 24 80 00 00 	mov    %r14,0x80(%rsp)
  401932:	00 
  401933:	4c 89 ac 24 88 00 00 	mov    %r13,0x88(%rsp)
  40193a:	00 
  40193b:	89 9c 24 a8 00 00 00 	mov    %ebx,0xa8(%rsp)
  401942:	44 89 a4 24 ac 00 00 	mov    %r12d,0xac(%rsp)
  401949:	00 
  40194a:	48 c7 44 24 70 00 00 	movq   $0x0,0x70(%rsp)
  401951:	00 00 
  401953:	e8 18 fe ff ff       	callq  401770 <GOMP_parallel_start@plt>
  401958:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  40195d:	e8 ae 03 00 00       	callq  401d10 <main._omp_fn.0>
  401962:	e8 a9 fc ff ff       	callq  401610 <GOMP_parallel_end@plt>
  401967:	c5 fb 10 44 24 70    	vmovsd 0x70(%rsp),%xmm0
  40196d:	83 6c 24 40 01       	subl   $0x1,0x40(%rsp)
  401972:	4c 8b 54 24 60       	mov    0x60(%rsp),%r10
  401977:	4c 8b 5c 24 68       	mov    0x68(%rsp),%r11
  40197c:	c5 fb 5d 54 24 48    	vminsd 0x48(%rsp),%xmm0,%xmm2
  401982:	4c 8b 7c 24 78       	mov    0x78(%rsp),%r15
  401987:	4c 8b b4 24 80 00 00 	mov    0x80(%rsp),%r14
  40198e:	00 
  40198f:	4c 8b ac 24 88 00 00 	mov    0x88(%rsp),%r13
  401996:	00 
  401997:	48 8b 8c 24 90 00 00 	mov    0x90(%rsp),%rcx
  40199e:	00 
  40199f:	4c 8b 84 24 98 00 00 	mov    0x98(%rsp),%r8
  4019a6:	00 
  4019a7:	4c 8b 8c 24 a0 00 00 	mov    0xa0(%rsp),%r9
  4019ae:	00 
  4019af:	8b 9c 24 a8 00 00 00 	mov    0xa8(%rsp),%ebx
  4019b6:	44 8b a4 24 ac 00 00 	mov    0xac(%rsp),%r12d
  4019bd:	00 
  4019be:	c5 fb 11 54 24 48    	vmovsd %xmm2,0x48(%rsp)
  4019c4:	0f 85 2e ff ff ff    	jne    4018f8 <main+0x178>
  4019ca:	c5 fb 10 25 a6 2a 00 	vmovsd 0x2aa6(%rip),%xmm4        # 404478 <__dso_handle+0xa0>
  4019d1:	00 
  4019d2:	89 d8                	mov    %ebx,%eax
  4019d4:	41 0f af c4          	imul   %r12d,%eax
  4019d8:	8b 74 24 38          	mov    0x38(%rsp),%esi
  4019dc:	bf e0 51 60 00       	mov    $0x6051e0,%edi
  4019e1:	c5 db 59 ca          	vmulsd %xmm2,%xmm4,%xmm1
  4019e5:	4c 89 4c 24 08       	mov    %r9,0x8(%rsp)
  4019ea:	4c 89 44 24 10       	mov    %r8,0x10(%rsp)
  4019ef:	48 89 4c 24 18       	mov    %rcx,0x18(%rsp)
  4019f4:	4c 89 5c 24 20       	mov    %r11,0x20(%rsp)
  4019f9:	8d 50 0f             	lea    0xf(%rax),%edx
  4019fc:	85 c0                	test   %eax,%eax
  4019fe:	4c 89 54 24 28       	mov    %r10,0x28(%rsp)
  401a03:	c5 f3 5e 0d 75 2a 00 	vdivsd 0x2a75(%rip),%xmm1,%xmm1        # 404480 <__dso_handle+0xa8>
  401a0a:	00 
  401a0b:	0f 48 c2             	cmovs  %edx,%eax
  401a0e:	c1 f8 04             	sar    $0x4,%eax
  401a11:	8d 04 40             	lea    (%rax,%rax,2),%eax
  401a14:	c1 e0 04             	shl    $0x4,%eax
  401a17:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  401a1b:	c5 fb 5e c1          	vdivsd %xmm1,%xmm0,%xmm0
  401a1f:	c5 fb 11 4c 24 30    	vmovsd %xmm1,0x30(%rsp)
  401a25:	c5 fb 11 44 24 40    	vmovsd %xmm0,0x40(%rsp)
  401a2b:	e8 20 fb ff ff       	callq  401550 <_ZNSolsEi@plt>
  401a30:	ba 01 00 00 00       	mov    $0x1,%edx
  401a35:	be f1 43 40 00       	mov    $0x4043f1,%esi
  401a3a:	48 89 c7             	mov    %rax,%rdi
  401a3d:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  401a42:	e8 d9 fc ff ff       	callq  401720 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
  401a47:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  401a4c:	c5 fb 10 44 24 40    	vmovsd 0x40(%rsp),%xmm0
  401a52:	e8 09 fc ff ff       	callq  401660 <_ZNSo9_M_insertIdEERSoT_@plt>
  401a57:	48 8d 74 24 5f       	lea    0x5f(%rsp),%rsi
  401a5c:	ba 01 00 00 00       	mov    $0x1,%edx
  401a61:	48 89 c7             	mov    %rax,%rdi
  401a64:	c6 44 24 5f 2c       	movb   $0x2c,0x5f(%rsp)
  401a69:	e8 b2 fc ff ff       	callq  401720 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
  401a6e:	c5 fb 10 4c 24 30    	vmovsd 0x30(%rsp),%xmm1
  401a74:	48 89 c7             	mov    %rax,%rdi
  401a77:	c5 f9 28 c1          	vmovapd %xmm1,%xmm0
  401a7b:	e8 e0 fb ff ff       	callq  401660 <_ZNSo9_M_insertIdEERSoT_@plt>
  401a80:	48 89 c2             	mov    %rax,%rdx
  401a83:	48 8b 00             	mov    (%rax),%rax
  401a86:	4c 8b 54 24 28       	mov    0x28(%rsp),%r10
  401a8b:	4c 8b 5c 24 20       	mov    0x20(%rsp),%r11
  401a90:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
  401a95:	4c 8b 44 24 10       	mov    0x10(%rsp),%r8
  401a9a:	4c 8b 4c 24 08       	mov    0x8(%rsp),%r9
  401a9f:	48 8b 40 e8          	mov    -0x18(%rax),%rax
  401aa3:	48 8b 84 02 f0 00 00 	mov    0xf0(%rdx,%rax,1),%rax
  401aaa:	00 
  401aab:	48 85 c0             	test   %rax,%rax
  401aae:	0f 84 47 01 00 00    	je     401bfb <main+0x47b>
  401ab4:	80 78 38 00          	cmpb   $0x0,0x38(%rax)
  401ab8:	0f 84 dc 00 00 00    	je     401b9a <main+0x41a>
  401abe:	0f b6 40 43          	movzbl 0x43(%rax),%eax
  401ac2:	48 89 d7             	mov    %rdx,%rdi
  401ac5:	0f be f0             	movsbl %al,%esi
  401ac8:	4c 89 4c 24 20       	mov    %r9,0x20(%rsp)
  401acd:	4c 89 44 24 28       	mov    %r8,0x28(%rsp)
  401ad2:	48 89 4c 24 30       	mov    %rcx,0x30(%rsp)
  401ad7:	4c 89 5c 24 40       	mov    %r11,0x40(%rsp)
  401adc:	4c 89 54 24 48       	mov    %r10,0x48(%rsp)
  401ae1:	e8 fa fa ff ff       	callq  4015e0 <_ZNSo3putEc@plt>
  401ae6:	48 89 c7             	mov    %rax,%rdi
  401ae9:	e8 c2 fa ff ff       	callq  4015b0 <_ZNSo5flushEv@plt>
  401aee:	83 44 24 38 01       	addl   $0x1,0x38(%rsp)
  401af3:	4c 8b 54 24 48       	mov    0x48(%rsp),%r10
  401af8:	83 7c 24 38 19       	cmpl   $0x19,0x38(%rsp)
  401afd:	4c 8b 5c 24 40       	mov    0x40(%rsp),%r11
  401b02:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
  401b07:	4c 8b 44 24 28       	mov    0x28(%rsp),%r8
  401b0c:	4c 8b 4c 24 20       	mov    0x20(%rsp),%r9
  401b11:	0f 85 89 fd ff ff    	jne    4018a0 <main+0x120>
  401b17:	48 8b 35 82 36 20 00 	mov    0x203682(%rip),%rsi        # 6051a0 <home>
  401b1e:	48 8d bc 24 30 01 00 	lea    0x130(%rsp),%rdi
  401b25:	00 
  401b26:	e8 d5 fb ff ff       	callq  401700 <stpcpy@plt>
  401b2b:	48 ba 72 65 73 75 6c 	movabs $0x2f73746c75736572,%rdx
  401b32:	74 73 2f 
  401b35:	48 8d 8c 24 30 01 00 	lea    0x130(%rsp),%rcx
  401b3c:	00 
  401b3d:	4c 89 f6             	mov    %r14,%rsi
  401b40:	48 89 10             	mov    %rdx,(%rax)
  401b43:	48 ba 6e 6e 2f 6d 79 	movabs $0x78382f796d2f6e6e,%rdx
  401b4a:	2f 38 78 
  401b4d:	4c 89 ff             	mov    %r15,%rdi
  401b50:	48 89 50 08          	mov    %rdx,0x8(%rax)
  401b54:	48 ba 38 2d 31 36 78 	movabs $0x2e36317836312d38,%rdx
  401b5b:	31 36 2e 
  401b5e:	c7 40 18 6a 70 67 00 	movl   $0x67706a,0x18(%rax)
  401b65:	48 89 50 10          	mov    %rdx,0x10(%rax)
  401b69:	4c 89 ea             	mov    %r13,%rdx
  401b6c:	e8 1f 04 00 00       	callq  401f90 <_Z11encodeImagePfS_S_Pc>
  401b71:	4c 89 ff             	mov    %r15,%rdi
  401b74:	e8 b7 f9 ff ff       	callq  401530 <free@plt>
  401b79:	4c 89 f7             	mov    %r14,%rdi
  401b7c:	e8 af f9 ff ff       	callq  401530 <free@plt>
  401b81:	4c 89 ef             	mov    %r13,%rdi
  401b84:	e8 a7 f9 ff ff       	callq  401530 <free@plt>
  401b89:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
  401b8d:	31 c0                	xor    %eax,%eax
  401b8f:	5b                   	pop    %rbx
  401b90:	41 5c                	pop    %r12
  401b92:	41 5d                	pop    %r13
  401b94:	41 5e                	pop    %r14
  401b96:	41 5f                	pop    %r15
  401b98:	5d                   	pop    %rbp
  401b99:	c3                   	retq   
  401b9a:	48 89 c7             	mov    %rax,%rdi
  401b9d:	4c 89 4c 24 18       	mov    %r9,0x18(%rsp)
  401ba2:	4c 89 44 24 20       	mov    %r8,0x20(%rsp)
  401ba7:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  401bac:	4c 89 5c 24 30       	mov    %r11,0x30(%rsp)
  401bb1:	4c 89 54 24 40       	mov    %r10,0x40(%rsp)
  401bb6:	48 89 54 24 10       	mov    %rdx,0x10(%rsp)
  401bbb:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  401bc0:	e8 db f9 ff ff       	callq  4015a0 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>
  401bc5:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  401bca:	be 0a 00 00 00       	mov    $0xa,%esi
  401bcf:	48 8b 10             	mov    (%rax),%rdx
  401bd2:	48 89 c7             	mov    %rax,%rdi
  401bd5:	ff 52 30             	callq  *0x30(%rdx)
  401bd8:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  401bdd:	4c 8b 4c 24 18       	mov    0x18(%rsp),%r9
  401be2:	4c 8b 44 24 20       	mov    0x20(%rsp),%r8
  401be7:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
  401bec:	4c 8b 5c 24 30       	mov    0x30(%rsp),%r11
  401bf1:	4c 8b 54 24 40       	mov    0x40(%rsp),%r10
  401bf6:	e9 c7 fe ff ff       	jmpq   401ac2 <main+0x342>
  401bfb:	e8 60 f9 ff ff       	callq  401560 <_ZSt16__throw_bad_castv@plt>

0000000000401c00 <_GLOBAL__sub_I_home>:
  401c00:	48 83 ec 08          	sub    $0x8,%rsp
  401c04:	bf f1 52 60 00       	mov    $0x6052f1,%edi
  401c09:	e8 22 fb ff ff       	callq  401730 <_ZNSt8ios_base4InitC1Ev@plt>
  401c0e:	ba d8 43 40 00       	mov    $0x4043d8,%edx
  401c13:	be f1 52 60 00       	mov    $0x6052f1,%esi
  401c18:	bf 30 16 40 00       	mov    $0x401630,%edi
  401c1d:	48 83 c4 08          	add    $0x8,%rsp
  401c21:	e9 8a fa ff ff       	jmpq   4016b0 <__cxa_atexit@plt>

0000000000401c26 <_start>:
  401c26:	31 ed                	xor    %ebp,%ebp
  401c28:	49 89 d1             	mov    %rdx,%r9
  401c2b:	5e                   	pop    %rsi
  401c2c:	48 89 e2             	mov    %rsp,%rdx
  401c2f:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  401c33:	50                   	push   %rax
  401c34:	54                   	push   %rsp
  401c35:	49 c7 c0 c0 43 40 00 	mov    $0x4043c0,%r8
  401c3c:	48 c7 c1 50 43 40 00 	mov    $0x404350,%rcx
  401c43:	48 c7 c7 80 17 40 00 	mov    $0x401780,%rdi
  401c4a:	e8 a1 fa ff ff       	callq  4016f0 <__libc_start_main@plt>
  401c4f:	f4                   	hlt    

0000000000401c50 <deregister_tm_clones>:
  401c50:	b8 af 51 60 00       	mov    $0x6051af,%eax
  401c55:	55                   	push   %rbp
  401c56:	48 2d a8 51 60 00    	sub    $0x6051a8,%rax
  401c5c:	48 83 f8 0e          	cmp    $0xe,%rax
  401c60:	48 89 e5             	mov    %rsp,%rbp
  401c63:	77 02                	ja     401c67 <deregister_tm_clones+0x17>
  401c65:	5d                   	pop    %rbp
  401c66:	c3                   	retq   
  401c67:	b8 00 00 00 00       	mov    $0x0,%eax
  401c6c:	48 85 c0             	test   %rax,%rax
  401c6f:	74 f4                	je     401c65 <deregister_tm_clones+0x15>
  401c71:	5d                   	pop    %rbp
  401c72:	bf a8 51 60 00       	mov    $0x6051a8,%edi
  401c77:	ff e0                	jmpq   *%rax
  401c79:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401c80 <register_tm_clones>:
  401c80:	b8 a8 51 60 00       	mov    $0x6051a8,%eax
  401c85:	55                   	push   %rbp
  401c86:	48 2d a8 51 60 00    	sub    $0x6051a8,%rax
  401c8c:	48 c1 f8 03          	sar    $0x3,%rax
  401c90:	48 89 e5             	mov    %rsp,%rbp
  401c93:	48 89 c2             	mov    %rax,%rdx
  401c96:	48 c1 ea 3f          	shr    $0x3f,%rdx
  401c9a:	48 01 d0             	add    %rdx,%rax
  401c9d:	48 d1 f8             	sar    %rax
  401ca0:	75 02                	jne    401ca4 <register_tm_clones+0x24>
  401ca2:	5d                   	pop    %rbp
  401ca3:	c3                   	retq   
  401ca4:	ba 00 00 00 00       	mov    $0x0,%edx
  401ca9:	48 85 d2             	test   %rdx,%rdx
  401cac:	74 f4                	je     401ca2 <register_tm_clones+0x22>
  401cae:	5d                   	pop    %rbp
  401caf:	48 89 c6             	mov    %rax,%rsi
  401cb2:	bf a8 51 60 00       	mov    $0x6051a8,%edi
  401cb7:	ff e2                	jmpq   *%rdx
  401cb9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401cc0 <__do_global_dtors_aux>:
  401cc0:	80 3d 29 36 20 00 00 	cmpb   $0x0,0x203629(%rip)        # 6052f0 <completed.6355>
  401cc7:	75 11                	jne    401cda <__do_global_dtors_aux+0x1a>
  401cc9:	55                   	push   %rbp
  401cca:	48 89 e5             	mov    %rsp,%rbp
  401ccd:	e8 7e ff ff ff       	callq  401c50 <deregister_tm_clones>
  401cd2:	5d                   	pop    %rbp
  401cd3:	c6 05 16 36 20 00 01 	movb   $0x1,0x203616(%rip)        # 6052f0 <completed.6355>
  401cda:	f3 c3                	repz retq 
  401cdc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401ce0 <frame_dummy>:
  401ce0:	48 83 3d e8 2f 20 00 	cmpq   $0x0,0x202fe8(%rip)        # 604cd0 <__JCR_END__>
  401ce7:	00 
  401ce8:	74 1e                	je     401d08 <frame_dummy+0x28>
  401cea:	b8 00 00 00 00       	mov    $0x0,%eax
  401cef:	48 85 c0             	test   %rax,%rax
  401cf2:	74 14                	je     401d08 <frame_dummy+0x28>
  401cf4:	55                   	push   %rbp
  401cf5:	bf d0 4c 60 00       	mov    $0x604cd0,%edi
  401cfa:	48 89 e5             	mov    %rsp,%rbp
  401cfd:	ff d0                	callq  *%rax
  401cff:	5d                   	pop    %rbp
  401d00:	e9 7b ff ff ff       	jmpq   401c80 <register_tm_clones>
  401d05:	0f 1f 00             	nopl   (%rax)
  401d08:	e9 73 ff ff ff       	jmpq   401c80 <register_tm_clones>
  401d0d:	0f 1f 00             	nopl   (%rax)

0000000000401d10 <main._omp_fn.0>:
  401d10:	41 57                	push   %r15
  401d12:	41 56                	push   %r14
  401d14:	41 55                	push   %r13
  401d16:	41 54                	push   %r12
  401d18:	55                   	push   %rbp
  401d19:	48 89 fd             	mov    %rdi,%rbp
  401d1c:	53                   	push   %rbx
  401d1d:	48 83 ec 28          	sub    $0x28,%rsp
  401d21:	8b 5f 48             	mov    0x48(%rdi),%ebx
  401d24:	8b 47 4c             	mov    0x4c(%rdi),%eax
  401d27:	0f af c3             	imul   %ebx,%eax
  401d2a:	44 8d 60 0f          	lea    0xf(%rax),%r12d
  401d2e:	85 c0                	test   %eax,%eax
  401d30:	44 0f 49 e0          	cmovns %eax,%r12d
  401d34:	41 c1 fc 04          	sar    $0x4,%r12d
  401d38:	e8 43 f8 ff ff       	callq  401580 <omp_get_num_threads@plt>
  401d3d:	41 89 c5             	mov    %eax,%r13d
  401d40:	e8 4b f8 ff ff       	callq  401590 <omp_get_thread_num@plt>
  401d45:	89 c1                	mov    %eax,%ecx
  401d47:	44 89 e0             	mov    %r12d,%eax
  401d4a:	99                   	cltd   
  401d4b:	41 f7 fd             	idiv   %r13d
  401d4e:	39 d1                	cmp    %edx,%ecx
  401d50:	7d 05                	jge    401d57 <main._omp_fn.0+0x47>
  401d52:	83 c0 01             	add    $0x1,%eax
  401d55:	31 d2                	xor    %edx,%edx
  401d57:	0f af c8             	imul   %eax,%ecx
  401d5a:	01 ca                	add    %ecx,%edx
  401d5c:	44 8d 24 02          	lea    (%rdx,%rax,1),%r12d
  401d60:	44 39 e2             	cmp    %r12d,%edx
  401d63:	0f 8d 81 01 00 00    	jge    401eea <main._omp_fn.0+0x1da>
  401d69:	89 d8                	mov    %ebx,%eax
  401d6b:	41 29 d4             	sub    %edx,%r12d
  401d6e:	44 8d 04 95 00 00 00 	lea    0x0(,%rdx,4),%r8d
  401d75:	00 
  401d76:	c1 e8 1f             	shr    $0x1f,%eax
  401d79:	4c 63 cb             	movslq %ebx,%r9
  401d7c:	4c 8b 75 28          	mov    0x28(%rbp),%r14
  401d80:	01 d8                	add    %ebx,%eax
  401d82:	49 63 f8             	movslq %r8d,%rdi
  401d85:	49 c1 e1 02          	shl    $0x2,%r9
  401d89:	89 c6                	mov    %eax,%esi
  401d8b:	48 8b 45 18          	mov    0x18(%rbp),%rax
  401d8f:	48 c1 e7 02          	shl    $0x2,%rdi
  401d93:	d1 fe                	sar    %esi
  401d95:	49 89 fb             	mov    %rdi,%r11
  401d98:	4c 8b 6d 20          	mov    0x20(%rbp),%r13
  401d9c:	c5 fb 10 6d 10       	vmovsd 0x10(%rbp),%xmm5
  401da1:	4c 03 5d 30          	add    0x30(%rbp),%r11
  401da5:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  401daa:	41 8d 44 24 ff       	lea    -0x1(%r12),%eax
  401daf:	48 89 6c 24 18       	mov    %rbp,0x18(%rsp)
  401db4:	49 89 fc             	mov    %rdi,%r12
  401db7:	4d 8d 51 10          	lea    0x10(%r9),%r10
  401dbb:	4c 03 65 38          	add    0x38(%rbp),%r12
  401dbf:	48 83 c0 01          	add    $0x1,%rax
  401dc3:	48 03 7d 40          	add    0x40(%rbp),%rdi
  401dc7:	31 c9                	xor    %ecx,%ecx
  401dc9:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  401dce:	89 f5                	mov    %esi,%ebp
  401dd0:	48 c1 64 24 08 04    	shlq   $0x4,0x8(%rsp)
  401dd6:	eb 24                	jmp    401dfc <main._omp_fn.0+0xec>
  401dd8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401ddf:	00 
  401de0:	c4 e1 fb 2a c0       	vcvtsi2sd %rax,%xmm0,%xmm0
  401de5:	41 83 c0 04          	add    $0x4,%r8d
  401de9:	48 83 c1 10          	add    $0x10,%rcx
  401ded:	48 3b 4c 24 08       	cmp    0x8(%rsp),%rcx
  401df2:	c5 d3 58 e8          	vaddsd %xmm0,%xmm5,%xmm5
  401df6:	0f 84 dc 00 00 00    	je     401ed8 <main._omp_fn.0+0x1c8>
  401dfc:	0f 31                	rdtsc  
  401dfe:	48 89 d6             	mov    %rdx,%rsi
  401e01:	89 c0                	mov    %eax,%eax
  401e03:	48 c1 e6 20          	shl    $0x20,%rsi
  401e07:	c4 c1 78 28 24 0b    	vmovaps (%r11,%rcx,1),%xmm4
  401e0d:	48 09 c6             	or     %rax,%rsi
  401e10:	44 89 c0             	mov    %r8d,%eax
  401e13:	99                   	cltd   
  401e14:	c4 e3 79 04 d4 50    	vpermilps $0x50,%xmm4,%xmm2
  401e1a:	f7 fd                	idiv   %ebp
  401e1c:	c4 c1 78 28 04 0c    	vmovaps (%r12,%rcx,1),%xmm0
  401e22:	c4 e3 79 04 e4 fa    	vpermilps $0xfa,%xmm4,%xmm4
  401e28:	c4 e3 79 04 c8 50    	vpermilps $0x50,%xmm0,%xmm1
  401e2e:	c4 e3 79 04 c0 fa    	vpermilps $0xfa,%xmm0,%xmm0
  401e34:	01 c0                	add    %eax,%eax
  401e36:	0f af c3             	imul   %ebx,%eax
  401e39:	44 8d 3c 50          	lea    (%rax,%rdx,2),%r15d
  401e3d:	4d 63 ff             	movslq %r15d,%r15
  401e40:	49 c1 e7 02          	shl    $0x2,%r15
  401e44:	4b 8d 14 3e          	lea    (%r14,%r15,1),%rdx
  401e48:	4b 8d 44 3d 00       	lea    0x0(%r13,%r15,1),%rax
  401e4d:	4c 03 7c 24 10       	add    0x10(%rsp),%r15
  401e52:	c4 c1 78 29 17       	vmovaps %xmm2,(%r15)
  401e57:	c4 81 78 29 14 0f    	vmovaps %xmm2,(%r15,%r9,1)
  401e5d:	c5 f8 28 14 0f       	vmovaps (%rdi,%rcx,1),%xmm2
  401e62:	c4 e3 79 04 da 50    	vpermilps $0x50,%xmm2,%xmm3
  401e68:	c4 e3 79 04 d2 fa    	vpermilps $0xfa,%xmm2,%xmm2
  401e6e:	c4 c1 78 29 67 10    	vmovaps %xmm4,0x10(%r15)
  401e74:	c4 81 78 29 24 17    	vmovaps %xmm4,(%r15,%r10,1)
  401e7a:	c5 f8 29 1a          	vmovaps %xmm3,(%rdx)
  401e7e:	c4 a1 78 29 1c 0a    	vmovaps %xmm3,(%rdx,%r9,1)
  401e84:	c5 f8 29 52 10       	vmovaps %xmm2,0x10(%rdx)
  401e89:	c4 a1 78 29 14 12    	vmovaps %xmm2,(%rdx,%r10,1)
  401e8f:	c5 f8 29 08          	vmovaps %xmm1,(%rax)
  401e93:	c4 a1 78 29 0c 08    	vmovaps %xmm1,(%rax,%r9,1)
  401e99:	c5 f8 29 40 10       	vmovaps %xmm0,0x10(%rax)
  401e9e:	c4 a1 78 29 04 10    	vmovaps %xmm0,(%rax,%r10,1)
  401ea4:	0f 31                	rdtsc  
  401ea6:	48 c1 e2 20          	shl    $0x20,%rdx
  401eaa:	89 c0                	mov    %eax,%eax
  401eac:	48 09 c2             	or     %rax,%rdx
  401eaf:	48 89 d0             	mov    %rdx,%rax
  401eb2:	48 29 f0             	sub    %rsi,%rax
  401eb5:	0f 89 25 ff ff ff    	jns    401de0 <main._omp_fn.0+0xd0>
  401ebb:	49 89 c7             	mov    %rax,%r15
  401ebe:	83 e0 01             	and    $0x1,%eax
  401ec1:	49 d1 ef             	shr    %r15
  401ec4:	49 09 c7             	or     %rax,%r15
  401ec7:	c4 c1 fb 2a c7       	vcvtsi2sd %r15,%xmm0,%xmm0
  401ecc:	c5 fb 58 c0          	vaddsd %xmm0,%xmm0,%xmm0
  401ed0:	e9 10 ff ff ff       	jmpq   401de5 <main._omp_fn.0+0xd5>
  401ed5:	0f 1f 00             	nopl   (%rax)
  401ed8:	48 8b 6c 24 18       	mov    0x18(%rsp),%rbp
  401edd:	c5 fb 11 6d 10       	vmovsd %xmm5,0x10(%rbp)
  401ee2:	48 89 75 00          	mov    %rsi,0x0(%rbp)
  401ee6:	48 89 55 08          	mov    %rdx,0x8(%rbp)
  401eea:	48 83 c4 28          	add    $0x28,%rsp
  401eee:	5b                   	pop    %rbx
  401eef:	5d                   	pop    %rbp
  401ef0:	41 5c                	pop    %r12
  401ef2:	41 5d                	pop    %r13
  401ef4:	41 5e                	pop    %r14
  401ef6:	41 5f                	pop    %r15
  401ef8:	c3                   	retq   
  401ef9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401f00 <_Z6kernelPfS_S_S_S_S_i>:
  401f00:	c5 f8 28 27          	vmovaps (%rdi),%xmm4
  401f04:	48 63 44 24 08       	movslq 0x8(%rsp),%rax
  401f09:	c4 e3 79 04 d4 50    	vpermilps $0x50,%xmm4,%xmm2
  401f0f:	c5 f8 28 06          	vmovaps (%rsi),%xmm0
  401f13:	c4 e3 79 04 e4 fa    	vpermilps $0xfa,%xmm4,%xmm4
  401f19:	c4 e3 79 04 c8 50    	vpermilps $0x50,%xmm0,%xmm1
  401f1f:	c5 f8 29 11          	vmovaps %xmm2,(%rcx)
  401f23:	c4 e3 79 04 c0 fa    	vpermilps $0xfa,%xmm0,%xmm0
  401f29:	48 8d 34 85 00 00 00 	lea    0x0(,%rax,4),%rsi
  401f30:	00 
  401f31:	c5 f8 29 14 81       	vmovaps %xmm2,(%rcx,%rax,4)
  401f36:	c5 f8 28 12          	vmovaps (%rdx),%xmm2
  401f3a:	c4 e3 79 04 da 50    	vpermilps $0x50,%xmm2,%xmm3
  401f40:	c4 e3 79 04 d2 fa    	vpermilps $0xfa,%xmm2,%xmm2
  401f46:	c5 f8 29 61 10       	vmovaps %xmm4,0x10(%rcx)
  401f4b:	c5 f8 29 64 31 10    	vmovaps %xmm4,0x10(%rcx,%rsi,1)
  401f51:	c4 c1 78 29 19       	vmovaps %xmm3,(%r9)
  401f56:	c4 c1 78 29 1c 81    	vmovaps %xmm3,(%r9,%rax,4)
  401f5c:	c4 c1 78 29 51 10    	vmovaps %xmm2,0x10(%r9)
  401f62:	c4 c1 78 29 54 31 10 	vmovaps %xmm2,0x10(%r9,%rsi,1)
  401f69:	c4 c1 78 29 08       	vmovaps %xmm1,(%r8)
  401f6e:	c4 c1 78 29 0c 80    	vmovaps %xmm1,(%r8,%rax,4)
  401f74:	c4 c1 78 29 40 10    	vmovaps %xmm0,0x10(%r8)
  401f7a:	c4 c1 78 29 44 30 10 	vmovaps %xmm0,0x10(%r8,%rsi,1)
  401f81:	c3                   	retq   
  401f82:	0f 1f 40 00          	nopl   0x0(%rax)
  401f86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401f8d:	00 00 00 

0000000000401f90 <_Z11encodeImagePfS_S_Pc>:
  401f90:	55                   	push   %rbp
  401f91:	c5 f1 ef c9          	vpxor  %xmm1,%xmm1,%xmm1
  401f95:	48 89 e5             	mov    %rsp,%rbp
  401f98:	41 54                	push   %r12
  401f9a:	53                   	push   %rbx
  401f9b:	48 89 cb             	mov    %rcx,%rbx
  401f9e:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  401fa2:	48 81 ec 70 02 00 00 	sub    $0x270,%rsp
  401fa9:	48 8d 84 24 98 00 00 	lea    0x98(%rsp),%rax
  401fb0:	00 
  401fb1:	c5 f9 6f 05 87 24 00 	vmovdqa 0x2487(%rip),%xmm0        # 404440 <__dso_handle+0x68>
  401fb8:	00 
  401fb9:	48 89 bc 24 60 01 00 	mov    %rdi,0x160(%rsp)
  401fc0:	00 
  401fc1:	48 89 bc 24 70 01 00 	mov    %rdi,0x170(%rsp)
  401fc8:	00 
  401fc9:	48 81 c7 00 10 00 00 	add    $0x1000,%rdi
  401fd0:	48 89 84 24 d0 00 00 	mov    %rax,0xd0(%rsp)
  401fd7:	00 
  401fd8:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
  401fdf:	00 
  401fe0:	48 c7 44 24 30 00 00 	movq   $0x0,0x30(%rsp)
  401fe7:	00 00 
  401fe9:	c5 f9 7f 8c 24 e0 00 	vmovdqa %xmm1,0xe0(%rsp)
  401ff0:	00 00 
  401ff2:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
  401ff9:	00 
  401ffa:	48 8d 84 24 f8 00 00 	lea    0xf8(%rsp),%rax
  402001:	00 
  402002:	48 c7 44 24 38 00 00 	movq   $0x0,0x38(%rsp)
  402009:	00 00 
  40200b:	c5 f9 7f 84 24 90 00 	vmovdqa %xmm0,0x90(%rsp)
  402012:	00 00 
  402014:	c5 f9 7f 8c 24 40 01 	vmovdqa %xmm1,0x140(%rsp)
  40201b:	00 00 
  40201d:	c5 f9 7f 84 24 f0 00 	vmovdqa %xmm0,0xf0(%rsp)
  402024:	00 00 
  402026:	48 89 84 24 30 01 00 	mov    %rax,0x130(%rsp)
  40202d:	00 
  40202e:	48 8d 84 24 40 01 00 	lea    0x140(%rsp),%rax
  402035:	00 
  402036:	48 c7 44 24 40 00 00 	movq   $0x0,0x40(%rsp)
  40203d:	00 00 
  40203f:	48 c7 84 24 c0 00 00 	movq   $0x0,0xc0(%rsp)
  402046:	00 00 00 00 00 
  40204b:	48 c7 84 24 b8 00 00 	movq   $0x0,0xb8(%rsp)
  402052:	00 00 00 00 00 
  402057:	48 89 84 24 38 01 00 	mov    %rax,0x138(%rsp)
  40205e:	00 
  40205f:	48 8d 84 24 58 01 00 	lea    0x158(%rsp),%rax
  402066:	00 
  402067:	48 c7 84 24 b0 00 00 	movq   $0x0,0xb0(%rsp)
  40206e:	00 00 00 00 00 
  402073:	48 c7 84 24 a0 00 00 	movq   $0x0,0xa0(%rsp)
  40207a:	00 00 00 00 00 
  40207f:	48 c7 84 24 a8 00 00 	movq   $0x0,0xa8(%rsp)
  402086:	00 00 00 00 00 
  40208b:	48 89 84 24 90 01 00 	mov    %rax,0x190(%rsp)
  402092:	00 
  402093:	48 8d 84 24 a0 01 00 	lea    0x1a0(%rsp),%rax
  40209a:	00 
  40209b:	48 c7 84 24 c8 00 00 	movq   $0x0,0xc8(%rsp)
  4020a2:	00 00 00 00 00 
  4020a7:	48 c7 84 24 20 01 00 	movq   $0x0,0x120(%rsp)
  4020ae:	00 00 00 00 00 
  4020b3:	48 c7 84 24 18 01 00 	movq   $0x0,0x118(%rsp)
  4020ba:	00 00 00 00 00 
  4020bf:	48 c7 84 24 10 01 00 	movq   $0x0,0x110(%rsp)
  4020c6:	00 00 00 00 00 
  4020cb:	48 c7 84 24 00 01 00 	movq   $0x0,0x100(%rsp)
  4020d2:	00 00 00 00 00 
  4020d7:	48 c7 84 24 08 01 00 	movq   $0x0,0x108(%rsp)
  4020de:	00 00 00 00 00 
  4020e3:	48 c7 84 24 28 01 00 	movq   $0x0,0x128(%rsp)
  4020ea:	00 00 00 00 00 
  4020ef:	48 c7 84 24 68 01 00 	movq   $0x0,0x168(%rsp)
  4020f6:	00 00 00 00 00 
  4020fb:	48 c7 84 24 88 01 00 	movq   $0x0,0x188(%rsp)
  402102:	00 00 00 00 00 
  402107:	48 89 84 24 98 01 00 	mov    %rax,0x198(%rsp)
  40210e:	00 
  40210f:	48 8d 84 24 b8 01 00 	lea    0x1b8(%rsp),%rax
  402116:	00 
  402117:	48 89 94 24 20 02 00 	mov    %rdx,0x220(%rsp)
  40211e:	00 
  40211f:	48 89 94 24 30 02 00 	mov    %rdx,0x230(%rsp)
  402126:	00 
  402127:	48 81 c2 00 10 00 00 	add    $0x1000,%rdx
  40212e:	48 89 bc 24 80 01 00 	mov    %rdi,0x180(%rsp)
  402135:	00 
  402136:	48 89 84 24 f0 01 00 	mov    %rax,0x1f0(%rsp)
  40213d:	00 
  40213e:	48 8d 84 24 00 02 00 	lea    0x200(%rsp),%rax
  402145:	00 
  402146:	48 89 bc 24 78 01 00 	mov    %rdi,0x178(%rsp)
  40214d:	00 
  40214e:	c5 f9 6f 0d fa 22 00 	vmovdqa 0x22fa(%rip),%xmm1        # 404450 <__dso_handle+0x78>
  402155:	00 
  402156:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  40215b:	48 89 84 24 f8 01 00 	mov    %rax,0x1f8(%rsp)
  402162:	00 
  402163:	48 8d 84 24 18 02 00 	lea    0x218(%rsp),%rax
  40216a:	00 
  40216b:	48 89 b4 24 c0 01 00 	mov    %rsi,0x1c0(%rsp)
  402172:	00 
  402173:	c5 f9 6f 05 e5 22 00 	vmovdqa 0x22e5(%rip),%xmm0        # 404460 <__dso_handle+0x88>
  40217a:	00 
  40217b:	48 89 b4 24 d0 01 00 	mov    %rsi,0x1d0(%rsp)
  402182:	00 
  402183:	48 89 84 24 50 02 00 	mov    %rax,0x250(%rsp)
  40218a:	00 
  40218b:	48 81 c6 00 10 00 00 	add    $0x1000,%rsi
  402192:	48 8d 84 24 60 02 00 	lea    0x260(%rsp),%rax
  402199:	00 
  40219a:	48 89 94 24 40 02 00 	mov    %rdx,0x240(%rsp)
  4021a1:	00 
  4021a2:	48 89 94 24 38 02 00 	mov    %rdx,0x238(%rsp)
  4021a9:	00 
  4021aa:	48 8d 94 24 10 02 00 	lea    0x210(%rsp),%rdx
  4021b1:	00 
  4021b2:	48 89 b4 24 e0 01 00 	mov    %rsi,0x1e0(%rsp)
  4021b9:	00 
  4021ba:	48 89 b4 24 d8 01 00 	mov    %rsi,0x1d8(%rsp)
  4021c1:	00 
  4021c2:	31 f6                	xor    %esi,%esi
  4021c4:	48 c7 84 24 c8 01 00 	movq   $0x0,0x1c8(%rsp)
  4021cb:	00 00 00 00 00 
  4021d0:	48 c7 84 24 e8 01 00 	movq   $0x0,0x1e8(%rsp)
  4021d7:	00 00 00 00 00 
  4021dc:	c5 f9 7f 8c 24 50 01 	vmovdqa %xmm1,0x150(%rsp)
  4021e3:	00 00 
  4021e5:	48 c7 84 24 28 02 00 	movq   $0x0,0x228(%rsp)
  4021ec:	00 00 00 00 00 
  4021f1:	48 c7 84 24 48 02 00 	movq   $0x0,0x248(%rsp)
  4021f8:	00 00 00 00 00 
  4021fd:	c5 f9 7f 84 24 a0 01 	vmovdqa %xmm0,0x1a0(%rsp)
  402204:	00 00 
  402206:	c5 f9 7f 8c 24 b0 01 	vmovdqa %xmm1,0x1b0(%rsp)
  40220d:	00 00 
  40220f:	c5 f9 7f 84 24 00 02 	vmovdqa %xmm0,0x200(%rsp)
  402216:	00 00 
  402218:	c5 f9 7f 8c 24 10 02 	vmovdqa %xmm1,0x210(%rsp)
  40221f:	00 00 
  402221:	c5 f9 7f 84 24 60 02 	vmovdqa %xmm0,0x260(%rsp)
  402228:	00 00 
  40222a:	48 89 84 24 58 02 00 	mov    %rax,0x258(%rsp)
  402231:	00 
  402232:	e8 59 11 00 00       	callq  403390 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_>
  402237:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  40223c:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
  402241:	48 39 c7             	cmp    %rax,%rdi
  402244:	0f 84 f7 05 00 00    	je     402841 <_Z11encodeImagePfS_S_Pc+0x8b1>
  40224a:	48 85 ff             	test   %rdi,%rdi
  40224d:	0f 84 73 05 00 00    	je     4027c6 <_Z11encodeImagePfS_S_Pc+0x836>
  402253:	c5 f9 6f 84 24 b0 01 	vmovdqa 0x1b0(%rsp),%xmm0
  40225a:	00 00 
  40225c:	c5 fa 7f 07          	vmovdqu %xmm0,(%rdi)
  402260:	48 8b 94 24 d0 01 00 	mov    0x1d0(%rsp),%rdx
  402267:	00 
  402268:	48 8b 84 24 c0 01 00 	mov    0x1c0(%rsp),%rax
  40226f:	00 
  402270:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  402274:	48 89 57 20          	mov    %rdx,0x20(%rdi)
  402278:	48 8b 94 24 d8 01 00 	mov    0x1d8(%rsp),%rdx
  40227f:	00 
  402280:	48 89 47 10          	mov    %rax,0x10(%rdi)
  402284:	48 8b 84 24 c8 01 00 	mov    0x1c8(%rsp),%rax
  40228b:	00 
  40228c:	c5 fa 7f 47 50       	vmovdqu %xmm0,0x50(%rdi)
  402291:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  402295:	48 8b 94 24 e0 01 00 	mov    0x1e0(%rsp),%rdx
  40229c:	00 
  40229d:	48 85 c0             	test   %rax,%rax
  4022a0:	48 89 57 30          	mov    %rdx,0x30(%rdi)
  4022a4:	48 8b 94 24 e8 01 00 	mov    0x1e8(%rsp),%rdx
  4022ab:	00 
  4022ac:	48 89 47 18          	mov    %rax,0x18(%rdi)
  4022b0:	48 89 57 38          	mov    %rdx,0x38(%rdi)
  4022b4:	48 8d 57 08          	lea    0x8(%rdi),%rdx
  4022b8:	48 89 57 40          	mov    %rdx,0x40(%rdi)
  4022bc:	48 8d 57 50          	lea    0x50(%rdi),%rdx
  4022c0:	48 89 57 48          	mov    %rdx,0x48(%rdi)
  4022c4:	74 04                	je     4022ca <_Z11encodeImagePfS_S_Pc+0x33a>
  4022c6:	f0 83 00 01          	lock addl $0x1,(%rax)
  4022ca:	83 bc 24 b4 01 00 00 	cmpl   $0x2,0x1b4(%rsp)
  4022d1:	02 
  4022d2:	0f 8f 88 04 00 00    	jg     402760 <_Z11encodeImagePfS_S_Pc+0x7d0>
  4022d8:	48 8b 94 24 f8 01 00 	mov    0x1f8(%rsp),%rdx
  4022df:	00 
  4022e0:	48 8b 47 48          	mov    0x48(%rdi),%rax
  4022e4:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  4022e9:	48 8b 12             	mov    (%rdx),%rdx
  4022ec:	48 89 10             	mov    %rdx,(%rax)
  4022ef:	48 8b 94 24 f8 01 00 	mov    0x1f8(%rsp),%rdx
  4022f6:	00 
  4022f7:	48 8b 47 48          	mov    0x48(%rdi),%rax
  4022fb:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  402300:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  402304:	48 89 50 08          	mov    %rdx,0x8(%rax)
  402308:	48 83 c7 60          	add    $0x60,%rdi
  40230c:	48 89 7c 24 38       	mov    %rdi,0x38(%rsp)
  402311:	48 39 fe             	cmp    %rdi,%rsi
  402314:	0f 84 10 05 00 00    	je     40282a <_Z11encodeImagePfS_S_Pc+0x89a>
  40231a:	48 85 ff             	test   %rdi,%rdi
  40231d:	0f 84 b0 00 00 00    	je     4023d3 <_Z11encodeImagePfS_S_Pc+0x443>
  402323:	c5 f9 6f 84 24 50 01 	vmovdqa 0x150(%rsp),%xmm0
  40232a:	00 00 
  40232c:	c5 fa 7f 07          	vmovdqu %xmm0,(%rdi)
  402330:	48 8b 94 24 70 01 00 	mov    0x170(%rsp),%rdx
  402337:	00 
  402338:	48 8b 84 24 60 01 00 	mov    0x160(%rsp),%rax
  40233f:	00 
  402340:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  402344:	48 89 57 20          	mov    %rdx,0x20(%rdi)
  402348:	48 8b 94 24 78 01 00 	mov    0x178(%rsp),%rdx
  40234f:	00 
  402350:	48 89 47 10          	mov    %rax,0x10(%rdi)
  402354:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  40235b:	00 
  40235c:	c5 fa 7f 47 50       	vmovdqu %xmm0,0x50(%rdi)
  402361:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  402365:	48 8b 94 24 80 01 00 	mov    0x180(%rsp),%rdx
  40236c:	00 
  40236d:	48 85 c0             	test   %rax,%rax
  402370:	48 89 57 30          	mov    %rdx,0x30(%rdi)
  402374:	48 8b 94 24 88 01 00 	mov    0x188(%rsp),%rdx
  40237b:	00 
  40237c:	48 89 47 18          	mov    %rax,0x18(%rdi)
  402380:	48 89 57 38          	mov    %rdx,0x38(%rdi)
  402384:	48 8d 57 08          	lea    0x8(%rdi),%rdx
  402388:	48 89 57 40          	mov    %rdx,0x40(%rdi)
  40238c:	48 8d 57 50          	lea    0x50(%rdi),%rdx
  402390:	48 89 57 48          	mov    %rdx,0x48(%rdi)
  402394:	74 04                	je     40239a <_Z11encodeImagePfS_S_Pc+0x40a>
  402396:	f0 83 00 01          	lock addl $0x1,(%rax)
  40239a:	83 bc 24 54 01 00 00 	cmpl   $0x2,0x154(%rsp)
  4023a1:	02 
  4023a2:	0f 8f 00 04 00 00    	jg     4027a8 <_Z11encodeImagePfS_S_Pc+0x818>
  4023a8:	48 8b 94 24 98 01 00 	mov    0x198(%rsp),%rdx
  4023af:	00 
  4023b0:	48 8b 47 48          	mov    0x48(%rdi),%rax
  4023b4:	48 8b 12             	mov    (%rdx),%rdx
  4023b7:	48 89 10             	mov    %rdx,(%rax)
  4023ba:	48 8b 94 24 98 01 00 	mov    0x198(%rsp),%rdx
  4023c1:	00 
  4023c2:	48 8b 47 48          	mov    0x48(%rdi),%rax
  4023c6:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  4023cb:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  4023cf:	48 89 50 08          	mov    %rdx,0x8(%rax)
  4023d3:	48 83 c7 60          	add    $0x60,%rdi
  4023d7:	48 89 7c 24 38       	mov    %rdi,0x38(%rsp)
  4023dc:	48 8d b4 24 90 00 00 	lea    0x90(%rsp),%rsi
  4023e3:	00 
  4023e4:	48 8d 7c 24 70       	lea    0x70(%rsp),%rdi
  4023e9:	e8 52 f1 ff ff       	callq  401540 <_ZN2cv12_OutputArrayC1ERNS_3MatE@plt>
  4023ee:	48 8d 74 24 70       	lea    0x70(%rsp),%rsi
  4023f3:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  4023f8:	e8 53 f2 ff ff       	callq  401650 <_ZN2cv5mergeERKSt6vectorINS_3MatESaIS1_EERKNS_12_OutputArrayE@plt>
  4023fd:	48 8d b4 24 90 00 00 	lea    0x90(%rsp),%rsi
  402404:	00 
  402405:	48 8d 7c 24 70       	lea    0x70(%rsp),%rdi
  40240a:	48 c7 44 24 50 00 00 	movq   $0x0,0x50(%rsp)
  402411:	00 00 
  402413:	48 c7 44 24 58 00 00 	movq   $0x0,0x58(%rsp)
  40241a:	00 00 
  40241c:	48 c7 44 24 60 00 00 	movq   $0x0,0x60(%rsp)
  402423:	00 00 
  402425:	e8 16 f2 ff ff       	callq  401640 <_ZN2cv11_InputArrayC1ERKNS_3MatE@plt>
  40242a:	48 8d 54 24 0e       	lea    0xe(%rsp),%rdx
  40242f:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  402434:	48 89 de             	mov    %rbx,%rsi
  402437:	e8 24 f3 ff ff       	callq  401760 <_ZNSsC1EPKcRKSaIcE@plt>
  40243c:	48 8d 54 24 50       	lea    0x50(%rsp),%rdx
  402441:	48 8d 74 24 70       	lea    0x70(%rsp),%rsi
  402446:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  40244b:	e8 b0 f1 ff ff       	callq  401600 <_ZN2cv7imwriteERKSsRKNS_11_InputArrayERKSt6vectorIiSaIiEE@plt>
  402450:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  402455:	48 8d 78 e8          	lea    -0x18(%rax),%rdi
  402459:	48 81 ff c0 51 60 00 	cmp    $0x6051c0,%rdi
  402460:	0f 85 ff 03 00 00    	jne    402865 <_Z11encodeImagePfS_S_Pc+0x8d5>
  402466:	48 8b 7c 24 50       	mov    0x50(%rsp),%rdi
  40246b:	48 85 ff             	test   %rdi,%rdi
  40246e:	74 05                	je     402475 <_Z11encodeImagePfS_S_Pc+0x4e5>
  402470:	e8 5b f1 ff ff       	callq  4015d0 <_ZdlPv@plt>
  402475:	48 8b 84 24 28 02 00 	mov    0x228(%rsp),%rax
  40247c:	00 
  40247d:	48 85 c0             	test   %rax,%rax
  402480:	74 0a                	je     40248c <_Z11encodeImagePfS_S_Pc+0x4fc>
  402482:	f0 83 28 01          	lock subl $0x1,(%rax)
  402486:	0f 84 44 03 00 00    	je     4027d0 <_Z11encodeImagePfS_S_Pc+0x840>
  40248c:	48 8b 84 24 50 02 00 	mov    0x250(%rsp),%rax
  402493:	00 
  402494:	48 8b bc 24 58 02 00 	mov    0x258(%rsp),%rdi
  40249b:	00 
  40249c:	48 c7 84 24 40 02 00 	movq   $0x0,0x240(%rsp)
  4024a3:	00 00 00 00 00 
  4024a8:	48 c7 84 24 38 02 00 	movq   $0x0,0x238(%rsp)
  4024af:	00 00 00 00 00 
  4024b4:	48 c7 84 24 30 02 00 	movq   $0x0,0x230(%rsp)
  4024bb:	00 00 00 00 00 
  4024c0:	48 c7 84 24 20 02 00 	movq   $0x0,0x220(%rsp)
  4024c7:	00 00 00 00 00 
  4024cc:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4024d2:	48 8d 84 24 60 02 00 	lea    0x260(%rsp),%rax
  4024d9:	00 
  4024da:	48 c7 84 24 28 02 00 	movq   $0x0,0x228(%rsp)
  4024e1:	00 00 00 00 00 
  4024e6:	48 39 c7             	cmp    %rax,%rdi
  4024e9:	74 05                	je     4024f0 <_Z11encodeImagePfS_S_Pc+0x560>
  4024eb:	e8 e0 f1 ff ff       	callq  4016d0 <_ZN2cv8fastFreeEPv@plt>
  4024f0:	48 8b 84 24 c8 01 00 	mov    0x1c8(%rsp),%rax
  4024f7:	00 
  4024f8:	48 85 c0             	test   %rax,%rax
  4024fb:	74 0a                	je     402507 <_Z11encodeImagePfS_S_Pc+0x577>
  4024fd:	f0 83 28 01          	lock subl $0x1,(%rax)
  402501:	0f 84 11 03 00 00    	je     402818 <_Z11encodeImagePfS_S_Pc+0x888>
  402507:	48 8b 84 24 f0 01 00 	mov    0x1f0(%rsp),%rax
  40250e:	00 
  40250f:	48 8b bc 24 f8 01 00 	mov    0x1f8(%rsp),%rdi
  402516:	00 
  402517:	48 c7 84 24 e0 01 00 	movq   $0x0,0x1e0(%rsp)
  40251e:	00 00 00 00 00 
  402523:	48 c7 84 24 d8 01 00 	movq   $0x0,0x1d8(%rsp)
  40252a:	00 00 00 00 00 
  40252f:	48 c7 84 24 d0 01 00 	movq   $0x0,0x1d0(%rsp)
  402536:	00 00 00 00 00 
  40253b:	48 c7 84 24 c0 01 00 	movq   $0x0,0x1c0(%rsp)
  402542:	00 00 00 00 00 
  402547:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40254d:	48 8d 84 24 00 02 00 	lea    0x200(%rsp),%rax
  402554:	00 
  402555:	48 c7 84 24 c8 01 00 	movq   $0x0,0x1c8(%rsp)
  40255c:	00 00 00 00 00 
  402561:	48 39 c7             	cmp    %rax,%rdi
  402564:	74 05                	je     40256b <_Z11encodeImagePfS_S_Pc+0x5db>
  402566:	e8 65 f1 ff ff       	callq  4016d0 <_ZN2cv8fastFreeEPv@plt>
  40256b:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  402572:	00 
  402573:	48 85 c0             	test   %rax,%rax
  402576:	74 0a                	je     402582 <_Z11encodeImagePfS_S_Pc+0x5f2>
  402578:	f0 83 28 01          	lock subl $0x1,(%rax)
  40257c:	0f 84 84 02 00 00    	je     402806 <_Z11encodeImagePfS_S_Pc+0x876>
  402582:	48 8b 84 24 90 01 00 	mov    0x190(%rsp),%rax
  402589:	00 
  40258a:	48 8b bc 24 98 01 00 	mov    0x198(%rsp),%rdi
  402591:	00 
  402592:	48 c7 84 24 80 01 00 	movq   $0x0,0x180(%rsp)
  402599:	00 00 00 00 00 
  40259e:	48 c7 84 24 78 01 00 	movq   $0x0,0x178(%rsp)
  4025a5:	00 00 00 00 00 
  4025aa:	48 c7 84 24 70 01 00 	movq   $0x0,0x170(%rsp)
  4025b1:	00 00 00 00 00 
  4025b6:	48 c7 84 24 60 01 00 	movq   $0x0,0x160(%rsp)
  4025bd:	00 00 00 00 00 
  4025c2:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4025c8:	48 8d 84 24 a0 01 00 	lea    0x1a0(%rsp),%rax
  4025cf:	00 
  4025d0:	48 c7 84 24 68 01 00 	movq   $0x0,0x168(%rsp)
  4025d7:	00 00 00 00 00 
  4025dc:	48 39 c7             	cmp    %rax,%rdi
  4025df:	74 05                	je     4025e6 <_Z11encodeImagePfS_S_Pc+0x656>
  4025e1:	e8 ea f0 ff ff       	callq  4016d0 <_ZN2cv8fastFreeEPv@plt>
  4025e6:	48 8b 84 24 08 01 00 	mov    0x108(%rsp),%rax
  4025ed:	00 
  4025ee:	48 85 c0             	test   %rax,%rax
  4025f1:	74 0a                	je     4025fd <_Z11encodeImagePfS_S_Pc+0x66d>
  4025f3:	f0 83 28 01          	lock subl $0x1,(%rax)
  4025f7:	0f 84 f7 01 00 00    	je     4027f4 <_Z11encodeImagePfS_S_Pc+0x864>
  4025fd:	48 8b 84 24 30 01 00 	mov    0x130(%rsp),%rax
  402604:	00 
  402605:	48 8b bc 24 38 01 00 	mov    0x138(%rsp),%rdi
  40260c:	00 
  40260d:	48 c7 84 24 20 01 00 	movq   $0x0,0x120(%rsp)
  402614:	00 00 00 00 00 
  402619:	48 c7 84 24 18 01 00 	movq   $0x0,0x118(%rsp)
  402620:	00 00 00 00 00 
  402625:	48 c7 84 24 10 01 00 	movq   $0x0,0x110(%rsp)
  40262c:	00 00 00 00 00 
  402631:	48 c7 84 24 00 01 00 	movq   $0x0,0x100(%rsp)
  402638:	00 00 00 00 00 
  40263d:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402643:	48 8d 84 24 40 01 00 	lea    0x140(%rsp),%rax
  40264a:	00 
  40264b:	48 c7 84 24 08 01 00 	movq   $0x0,0x108(%rsp)
  402652:	00 00 00 00 00 
  402657:	48 39 c7             	cmp    %rax,%rdi
  40265a:	74 05                	je     402661 <_Z11encodeImagePfS_S_Pc+0x6d1>
  40265c:	e8 6f f0 ff ff       	callq  4016d0 <_ZN2cv8fastFreeEPv@plt>
  402661:	48 8b 84 24 a8 00 00 	mov    0xa8(%rsp),%rax
  402668:	00 
  402669:	48 85 c0             	test   %rax,%rax
  40266c:	74 0a                	je     402678 <_Z11encodeImagePfS_S_Pc+0x6e8>
  40266e:	f0 83 28 01          	lock subl $0x1,(%rax)
  402672:	0f 84 6a 01 00 00    	je     4027e2 <_Z11encodeImagePfS_S_Pc+0x852>
  402678:	48 8b 84 24 d0 00 00 	mov    0xd0(%rsp),%rax
  40267f:	00 
  402680:	48 8b bc 24 d8 00 00 	mov    0xd8(%rsp),%rdi
  402687:	00 
  402688:	48 c7 84 24 c0 00 00 	movq   $0x0,0xc0(%rsp)
  40268f:	00 00 00 00 00 
  402694:	48 c7 84 24 b8 00 00 	movq   $0x0,0xb8(%rsp)
  40269b:	00 00 00 00 00 
  4026a0:	48 c7 84 24 b0 00 00 	movq   $0x0,0xb0(%rsp)
  4026a7:	00 00 00 00 00 
  4026ac:	48 c7 84 24 a0 00 00 	movq   $0x0,0xa0(%rsp)
  4026b3:	00 00 00 00 00 
  4026b8:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4026be:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
  4026c5:	00 
  4026c6:	48 c7 84 24 a8 00 00 	movq   $0x0,0xa8(%rsp)
  4026cd:	00 00 00 00 00 
  4026d2:	48 39 c7             	cmp    %rax,%rdi
  4026d5:	74 05                	je     4026dc <_Z11encodeImagePfS_S_Pc+0x74c>
  4026d7:	e8 f4 ef ff ff       	callq  4016d0 <_ZN2cv8fastFreeEPv@plt>
  4026dc:	4c 8b 64 24 38       	mov    0x38(%rsp),%r12
  4026e1:	48 8b 5c 24 30       	mov    0x30(%rsp),%rbx
  4026e6:	49 39 dc             	cmp    %rbx,%r12
  4026e9:	4c 89 e7             	mov    %r12,%rdi
  4026ec:	75 57                	jne    402745 <_Z11encodeImagePfS_S_Pc+0x7b5>
  4026ee:	e9 9a 00 00 00       	jmpq   40278d <_Z11encodeImagePfS_S_Pc+0x7fd>
  4026f3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4026f8:	48 8b 43 40          	mov    0x40(%rbx),%rax
  4026fc:	48 c7 43 30 00 00 00 	movq   $0x0,0x30(%rbx)
  402703:	00 
  402704:	48 c7 43 28 00 00 00 	movq   $0x0,0x28(%rbx)
  40270b:	00 
  40270c:	48 c7 43 20 00 00 00 	movq   $0x0,0x20(%rbx)
  402713:	00 
  402714:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
  40271b:	00 
  40271c:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402722:	48 8b 7b 48          	mov    0x48(%rbx),%rdi
  402726:	48 8d 43 50          	lea    0x50(%rbx),%rax
  40272a:	48 c7 43 18 00 00 00 	movq   $0x0,0x18(%rbx)
  402731:	00 
  402732:	48 39 c7             	cmp    %rax,%rdi
  402735:	74 05                	je     40273c <_Z11encodeImagePfS_S_Pc+0x7ac>
  402737:	e8 94 ef ff ff       	callq  4016d0 <_ZN2cv8fastFreeEPv@plt>
  40273c:	48 83 c3 60          	add    $0x60,%rbx
  402740:	49 39 dc             	cmp    %rbx,%r12
  402743:	74 43                	je     402788 <_Z11encodeImagePfS_S_Pc+0x7f8>
  402745:	48 8b 43 18          	mov    0x18(%rbx),%rax
  402749:	48 85 c0             	test   %rax,%rax
  40274c:	74 aa                	je     4026f8 <_Z11encodeImagePfS_S_Pc+0x768>
  40274e:	f0 83 28 01          	lock subl $0x1,(%rax)
  402752:	75 a4                	jne    4026f8 <_Z11encodeImagePfS_S_Pc+0x768>
  402754:	48 89 df             	mov    %rbx,%rdi
  402757:	e8 64 ee ff ff       	callq  4015c0 <_ZN2cv3Mat10deallocateEv@plt>
  40275c:	eb 9a                	jmp    4026f8 <_Z11encodeImagePfS_S_Pc+0x768>
  40275e:	66 90                	xchg   %ax,%ax
  402760:	48 8d b4 24 b0 01 00 	lea    0x1b0(%rsp),%rsi
  402767:	00 
  402768:	c7 47 04 00 00 00 00 	movl   $0x0,0x4(%rdi)
  40276f:	e8 8c ed ff ff       	callq  401500 <_ZN2cv3Mat8copySizeERKS0_@plt>
  402774:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  402779:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  40277e:	e9 85 fb ff ff       	jmpq   402308 <_Z11encodeImagePfS_S_Pc+0x378>
  402783:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  402788:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  40278d:	48 85 ff             	test   %rdi,%rdi
  402790:	74 05                	je     402797 <_Z11encodeImagePfS_S_Pc+0x807>
  402792:	e8 39 ee ff ff       	callq  4015d0 <_ZdlPv@plt>
  402797:	48 8d 65 f0          	lea    -0x10(%rbp),%rsp
  40279b:	31 c0                	xor    %eax,%eax
  40279d:	5b                   	pop    %rbx
  40279e:	41 5c                	pop    %r12
  4027a0:	5d                   	pop    %rbp
  4027a1:	c3                   	retq   
  4027a2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4027a8:	48 8d b4 24 50 01 00 	lea    0x150(%rsp),%rsi
  4027af:	00 
  4027b0:	c7 47 04 00 00 00 00 	movl   $0x0,0x4(%rdi)
  4027b7:	e8 44 ed ff ff       	callq  401500 <_ZN2cv3Mat8copySizeERKS0_@plt>
  4027bc:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  4027c1:	e9 0d fc ff ff       	jmpq   4023d3 <_Z11encodeImagePfS_S_Pc+0x443>
  4027c6:	48 89 c6             	mov    %rax,%rsi
  4027c9:	31 ff                	xor    %edi,%edi
  4027cb:	e9 38 fb ff ff       	jmpq   402308 <_Z11encodeImagePfS_S_Pc+0x378>
  4027d0:	48 8d bc 24 10 02 00 	lea    0x210(%rsp),%rdi
  4027d7:	00 
  4027d8:	e8 e3 ed ff ff       	callq  4015c0 <_ZN2cv3Mat10deallocateEv@plt>
  4027dd:	e9 aa fc ff ff       	jmpq   40248c <_Z11encodeImagePfS_S_Pc+0x4fc>
  4027e2:	48 8d bc 24 90 00 00 	lea    0x90(%rsp),%rdi
  4027e9:	00 
  4027ea:	e8 d1 ed ff ff       	callq  4015c0 <_ZN2cv3Mat10deallocateEv@plt>
  4027ef:	e9 84 fe ff ff       	jmpq   402678 <_Z11encodeImagePfS_S_Pc+0x6e8>
  4027f4:	48 8d bc 24 f0 00 00 	lea    0xf0(%rsp),%rdi
  4027fb:	00 
  4027fc:	e8 bf ed ff ff       	callq  4015c0 <_ZN2cv3Mat10deallocateEv@plt>
  402801:	e9 f7 fd ff ff       	jmpq   4025fd <_Z11encodeImagePfS_S_Pc+0x66d>
  402806:	48 8d bc 24 50 01 00 	lea    0x150(%rsp),%rdi
  40280d:	00 
  40280e:	e8 ad ed ff ff       	callq  4015c0 <_ZN2cv3Mat10deallocateEv@plt>
  402813:	e9 6a fd ff ff       	jmpq   402582 <_Z11encodeImagePfS_S_Pc+0x5f2>
  402818:	48 8d bc 24 b0 01 00 	lea    0x1b0(%rsp),%rdi
  40281f:	00 
  402820:	e8 9b ed ff ff       	callq  4015c0 <_ZN2cv3Mat10deallocateEv@plt>
  402825:	e9 dd fc ff ff       	jmpq   402507 <_Z11encodeImagePfS_S_Pc+0x577>
  40282a:	48 8d 94 24 50 01 00 	lea    0x150(%rsp),%rdx
  402831:	00 
  402832:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  402837:	e8 54 0b 00 00       	callq  403390 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_>
  40283c:	e9 9b fb ff ff       	jmpq   4023dc <_Z11encodeImagePfS_S_Pc+0x44c>
  402841:	48 89 fe             	mov    %rdi,%rsi
  402844:	48 8d 94 24 b0 01 00 	lea    0x1b0(%rsp),%rdx
  40284b:	00 
  40284c:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  402851:	e8 3a 0b 00 00       	callq  403390 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_>
  402856:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  40285b:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  402860:	e9 ac fa ff ff       	jmpq   402311 <_Z11encodeImagePfS_S_Pc+0x381>
  402865:	b9 e0 16 40 00       	mov    $0x4016e0,%ecx
  40286a:	48 8d 57 10          	lea    0x10(%rdi),%rdx
  40286e:	48 85 c9             	test   %rcx,%rcx
  402871:	74 52                	je     4028c5 <_Z11encodeImagePfS_S_Pc+0x935>
  402873:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402878:	f0 0f c1 02          	lock xadd %eax,(%rdx)
  40287c:	85 c0                	test   %eax,%eax
  40287e:	0f 8f e2 fb ff ff    	jg     402466 <_Z11encodeImagePfS_S_Pc+0x4d6>
  402884:	48 8d 74 24 0f       	lea    0xf(%rsp),%rsi
  402889:	e8 82 ee ff ff       	callq  401710 <_ZNSs4_Rep10_M_destroyERKSaIcE@plt>
  40288e:	e9 d3 fb ff ff       	jmpq   402466 <_Z11encodeImagePfS_S_Pc+0x4d6>
  402893:	48 89 c3             	mov    %rax,%rbx
  402896:	c5 f8 77             	vzeroupper 
  402899:	48 8d bc 24 f0 00 00 	lea    0xf0(%rsp),%rdi
  4028a0:	00 
  4028a1:	e8 8a 09 00 00       	callq  403230 <_ZN2cv3MatD1Ev>
  4028a6:	48 8d bc 24 90 00 00 	lea    0x90(%rsp),%rdi
  4028ad:	00 
  4028ae:	e8 7d 09 00 00       	callq  403230 <_ZN2cv3MatD1Ev>
  4028b3:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  4028b8:	e8 f3 09 00 00       	callq  4032b0 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev>
  4028bd:	48 89 df             	mov    %rbx,%rdi
  4028c0:	e8 5b ed ff ff       	callq  401620 <_Unwind_Resume@plt>
  4028c5:	8b 50 f8             	mov    -0x8(%rax),%edx
  4028c8:	8d 4a ff             	lea    -0x1(%rdx),%ecx
  4028cb:	89 48 f8             	mov    %ecx,-0x8(%rax)
  4028ce:	89 d0                	mov    %edx,%eax
  4028d0:	eb aa                	jmp    40287c <_Z11encodeImagePfS_S_Pc+0x8ec>
  4028d2:	48 89 c3             	mov    %rax,%rbx
  4028d5:	c5 f8 77             	vzeroupper 
  4028d8:	eb cc                	jmp    4028a6 <_Z11encodeImagePfS_S_Pc+0x916>
  4028da:	48 89 c3             	mov    %rax,%rbx
  4028dd:	c5 f8 77             	vzeroupper 
  4028e0:	48 8d bc 24 b0 01 00 	lea    0x1b0(%rsp),%rdi
  4028e7:	00 
  4028e8:	e8 43 09 00 00       	callq  403230 <_ZN2cv3MatD1Ev>
  4028ed:	48 8d bc 24 50 01 00 	lea    0x150(%rsp),%rdi
  4028f4:	00 
  4028f5:	e8 36 09 00 00       	callq  403230 <_ZN2cv3MatD1Ev>
  4028fa:	eb 9d                	jmp    402899 <_Z11encodeImagePfS_S_Pc+0x909>
  4028fc:	48 89 c3             	mov    %rax,%rbx
  4028ff:	c5 f8 77             	vzeroupper 
  402902:	eb e9                	jmp    4028ed <_Z11encodeImagePfS_S_Pc+0x95d>
  402904:	48 89 c3             	mov    %rax,%rbx
  402907:	c5 f8 77             	vzeroupper 
  40290a:	eb a7                	jmp    4028b3 <_Z11encodeImagePfS_S_Pc+0x923>
  40290c:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  402911:	48 89 c3             	mov    %rax,%rbx
  402914:	48 85 ff             	test   %rdi,%rdi
  402917:	74 1f                	je     402938 <_Z11encodeImagePfS_S_Pc+0x9a8>
  402919:	c5 f8 77             	vzeroupper 
  40291c:	e8 af ec ff ff       	callq  4015d0 <_ZdlPv@plt>
  402921:	eb 9a                	jmp    4028bd <_Z11encodeImagePfS_S_Pc+0x92d>
  402923:	48 89 c3             	mov    %rax,%rbx
  402926:	c5 f8 77             	vzeroupper 
  402929:	48 8d bc 24 10 02 00 	lea    0x210(%rsp),%rdi
  402930:	00 
  402931:	e8 fa 08 00 00       	callq  403230 <_ZN2cv3MatD1Ev>
  402936:	eb a8                	jmp    4028e0 <_Z11encodeImagePfS_S_Pc+0x950>
  402938:	c5 f8 77             	vzeroupper 
  40293b:	eb 80                	jmp    4028bd <_Z11encodeImagePfS_S_Pc+0x92d>
  40293d:	48 89 c3             	mov    %rax,%rbx
  402940:	c5 f8 77             	vzeroupper 
  402943:	48 8b 7c 24 50       	mov    0x50(%rsp),%rdi
  402948:	48 85 ff             	test   %rdi,%rdi
  40294b:	74 dc                	je     402929 <_Z11encodeImagePfS_S_Pc+0x999>
  40294d:	e8 7e ec ff ff       	callq  4015d0 <_ZdlPv@plt>
  402952:	eb d5                	jmp    402929 <_Z11encodeImagePfS_S_Pc+0x999>
  402954:	48 89 c3             	mov    %rax,%rbx
  402957:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40295c:	48 8d 74 24 0f       	lea    0xf(%rsp),%rsi
  402961:	48 8d 78 e8          	lea    -0x18(%rax),%rdi
  402965:	c5 f8 77             	vzeroupper 
  402968:	e8 13 ed ff ff       	callq  401680 <_ZNSs4_Rep10_M_disposeERKSaIcE@plt>
  40296d:	eb d4                	jmp    402943 <_Z11encodeImagePfS_S_Pc+0x9b3>
  40296f:	90                   	nop

0000000000402970 <_Z11decodeImagePfS_S_>:
  402970:	55                   	push   %rbp
  402971:	48 89 e5             	mov    %rsp,%rbp
  402974:	41 57                	push   %r15
  402976:	41 56                	push   %r14
  402978:	41 55                	push   %r13
  40297a:	49 89 d5             	mov    %rdx,%r13
  40297d:	41 54                	push   %r12
  40297f:	49 89 f4             	mov    %rsi,%r12
  402982:	be e0 43 40 00       	mov    $0x4043e0,%esi
  402987:	53                   	push   %rbx
  402988:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  40298c:	48 81 ec a0 02 00 00 	sub    $0x2a0,%rsp
  402993:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
  402998:	48 8d 9c 24 80 01 00 	lea    0x180(%rsp),%rbx
  40299f:	00 
  4029a0:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4029a5:	48 c7 84 24 b8 00 00 	movq   $0x0,0xb8(%rsp)
  4029ac:	00 00 00 00 00 
  4029b1:	48 c7 84 24 b0 00 00 	movq   $0x0,0xb0(%rsp)
  4029b8:	00 00 00 00 00 
  4029bd:	48 8d 54 24 1e       	lea    0x1e(%rsp),%rdx
  4029c2:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
  4029c9:	00 
  4029ca:	48 8d 84 24 b0 00 00 	lea    0xb0(%rsp),%rax
  4029d1:	00 
  4029d2:	c7 44 24 60 00 00 ff 	movl   $0x42ff0000,0x60(%rsp)
  4029d9:	42 
  4029da:	c7 44 24 6c 00 00 00 	movl   $0x0,0x6c(%rsp)
  4029e1:	00 
  4029e2:	c7 44 24 68 00 00 00 	movl   $0x0,0x68(%rsp)
  4029e9:	00 
  4029ea:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
  4029ef:	48 89 84 24 a8 00 00 	mov    %rax,0xa8(%rsp)
  4029f6:	00 
  4029f7:	48 8d 84 24 c8 00 00 	lea    0xc8(%rsp),%rax
  4029fe:	00 
  4029ff:	c7 44 24 64 00 00 00 	movl   $0x0,0x64(%rsp)
  402a06:	00 
  402a07:	48 c7 84 24 90 00 00 	movq   $0x0,0x90(%rsp)
  402a0e:	00 00 00 00 00 
  402a13:	48 c7 84 24 88 00 00 	movq   $0x0,0x88(%rsp)
  402a1a:	00 00 00 00 00 
  402a1f:	48 89 84 24 00 01 00 	mov    %rax,0x100(%rsp)
  402a26:	00 
  402a27:	48 8d 84 24 10 01 00 	lea    0x110(%rsp),%rax
  402a2e:	00 
  402a2f:	48 c7 84 24 80 00 00 	movq   $0x0,0x80(%rsp)
  402a36:	00 00 00 00 00 
  402a3b:	48 c7 44 24 70 00 00 	movq   $0x0,0x70(%rsp)
  402a42:	00 00 
  402a44:	48 c7 44 24 78 00 00 	movq   $0x0,0x78(%rsp)
  402a4b:	00 00 
  402a4d:	48 89 84 24 08 01 00 	mov    %rax,0x108(%rsp)
  402a54:	00 
  402a55:	48 8d 43 08          	lea    0x8(%rbx),%rax
  402a59:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
  402a60:	00 00 00 00 00 
  402a65:	48 c7 84 24 18 01 00 	movq   $0x0,0x118(%rsp)
  402a6c:	00 00 00 00 00 
  402a71:	48 c7 84 24 10 01 00 	movq   $0x0,0x110(%rsp)
  402a78:	00 00 00 00 00 
  402a7d:	c7 84 24 c0 00 00 00 	movl   $0x42ff0000,0xc0(%rsp)
  402a84:	00 00 ff 42 
  402a88:	c7 84 24 cc 00 00 00 	movl   $0x0,0xcc(%rsp)
  402a8f:	00 00 00 00 
  402a93:	c7 84 24 c8 00 00 00 	movl   $0x0,0xc8(%rsp)
  402a9a:	00 00 00 00 
  402a9e:	c7 84 24 c4 00 00 00 	movl   $0x0,0xc4(%rsp)
  402aa5:	00 00 00 00 
  402aa9:	48 c7 84 24 f0 00 00 	movq   $0x0,0xf0(%rsp)
  402ab0:	00 00 00 00 00 
  402ab5:	48 c7 84 24 e8 00 00 	movq   $0x0,0xe8(%rsp)
  402abc:	00 00 00 00 00 
  402ac1:	48 c7 84 24 e0 00 00 	movq   $0x0,0xe0(%rsp)
  402ac8:	00 00 00 00 00 
  402acd:	48 c7 84 24 d0 00 00 	movq   $0x0,0xd0(%rsp)
  402ad4:	00 00 00 00 00 
  402ad9:	48 89 84 24 c0 01 00 	mov    %rax,0x1c0(%rsp)
  402ae0:	00 
  402ae1:	48 8d 43 50          	lea    0x50(%rbx),%rax
  402ae5:	48 c7 84 24 d8 00 00 	movq   $0x0,0xd8(%rsp)
  402aec:	00 00 00 00 00 
  402af1:	48 c7 84 24 f8 00 00 	movq   $0x0,0xf8(%rsp)
  402af8:	00 00 00 00 00 
  402afd:	48 c7 84 24 d8 01 00 	movq   $0x0,0x1d8(%rsp)
  402b04:	00 00 00 00 00 
  402b09:	48 89 84 24 c8 01 00 	mov    %rax,0x1c8(%rsp)
  402b10:	00 
  402b11:	48 8d 43 68          	lea    0x68(%rbx),%rax
  402b15:	48 c7 84 24 d0 01 00 	movq   $0x0,0x1d0(%rsp)
  402b1c:	00 00 00 00 00 
  402b21:	c7 84 24 80 01 00 00 	movl   $0x42ff0000,0x180(%rsp)
  402b28:	00 00 ff 42 
  402b2c:	c7 84 24 8c 01 00 00 	movl   $0x0,0x18c(%rsp)
  402b33:	00 00 00 00 
  402b37:	48 89 84 24 20 02 00 	mov    %rax,0x220(%rsp)
  402b3e:	00 
  402b3f:	48 8d 83 b0 00 00 00 	lea    0xb0(%rbx),%rax
  402b46:	c7 84 24 88 01 00 00 	movl   $0x0,0x188(%rsp)
  402b4d:	00 00 00 00 
  402b51:	c7 84 24 84 01 00 00 	movl   $0x0,0x184(%rsp)
  402b58:	00 00 00 00 
  402b5c:	48 c7 84 24 b0 01 00 	movq   $0x0,0x1b0(%rsp)
  402b63:	00 00 00 00 00 
  402b68:	48 89 84 24 28 02 00 	mov    %rax,0x228(%rsp)
  402b6f:	00 
  402b70:	48 8d 83 c8 00 00 00 	lea    0xc8(%rbx),%rax
  402b77:	48 c7 84 24 a8 01 00 	movq   $0x0,0x1a8(%rsp)
  402b7e:	00 00 00 00 00 
  402b83:	48 c7 84 24 a0 01 00 	movq   $0x0,0x1a0(%rsp)
  402b8a:	00 00 00 00 00 
  402b8f:	48 c7 84 24 90 01 00 	movq   $0x0,0x190(%rsp)
  402b96:	00 00 00 00 00 
  402b9b:	48 89 84 24 80 02 00 	mov    %rax,0x280(%rsp)
  402ba2:	00 
  402ba3:	48 8d 83 10 01 00 00 	lea    0x110(%rbx),%rax
  402baa:	48 c7 84 24 98 01 00 	movq   $0x0,0x198(%rsp)
  402bb1:	00 00 00 00 00 
  402bb6:	48 c7 84 24 b8 01 00 	movq   $0x0,0x1b8(%rsp)
  402bbd:	00 00 00 00 00 
  402bc2:	48 c7 84 24 38 02 00 	movq   $0x0,0x238(%rsp)
  402bc9:	00 00 00 00 00 
  402bce:	48 c7 84 24 30 02 00 	movq   $0x0,0x230(%rsp)
  402bd5:	00 00 00 00 00 
  402bda:	c7 84 24 e0 01 00 00 	movl   $0x42ff0000,0x1e0(%rsp)
  402be1:	00 00 ff 42 
  402be5:	c7 84 24 ec 01 00 00 	movl   $0x0,0x1ec(%rsp)
  402bec:	00 00 00 00 
  402bf0:	c7 84 24 e8 01 00 00 	movl   $0x0,0x1e8(%rsp)
  402bf7:	00 00 00 00 
  402bfb:	c7 84 24 e4 01 00 00 	movl   $0x0,0x1e4(%rsp)
  402c02:	00 00 00 00 
  402c06:	48 c7 84 24 10 02 00 	movq   $0x0,0x210(%rsp)
  402c0d:	00 00 00 00 00 
  402c12:	48 c7 84 24 08 02 00 	movq   $0x0,0x208(%rsp)
  402c19:	00 00 00 00 00 
  402c1e:	48 c7 84 24 00 02 00 	movq   $0x0,0x200(%rsp)
  402c25:	00 00 00 00 00 
  402c2a:	48 c7 84 24 f0 01 00 	movq   $0x0,0x1f0(%rsp)
  402c31:	00 00 00 00 00 
  402c36:	48 c7 84 24 f8 01 00 	movq   $0x0,0x1f8(%rsp)
  402c3d:	00 00 00 00 00 
  402c42:	48 c7 84 24 18 02 00 	movq   $0x0,0x218(%rsp)
  402c49:	00 00 00 00 00 
  402c4e:	48 89 84 24 88 02 00 	mov    %rax,0x288(%rsp)
  402c55:	00 
  402c56:	48 c7 84 24 98 02 00 	movq   $0x0,0x298(%rsp)
  402c5d:	00 00 00 00 00 
  402c62:	48 c7 84 24 90 02 00 	movq   $0x0,0x290(%rsp)
  402c69:	00 00 00 00 00 
  402c6e:	c7 84 24 40 02 00 00 	movl   $0x42ff0000,0x240(%rsp)
  402c75:	00 00 ff 42 
  402c79:	c7 84 24 4c 02 00 00 	movl   $0x0,0x24c(%rsp)
  402c80:	00 00 00 00 
  402c84:	c7 84 24 48 02 00 00 	movl   $0x0,0x248(%rsp)
  402c8b:	00 00 00 00 
  402c8f:	c7 84 24 44 02 00 00 	movl   $0x0,0x244(%rsp)
  402c96:	00 00 00 00 
  402c9a:	48 c7 84 24 70 02 00 	movq   $0x0,0x270(%rsp)
  402ca1:	00 00 00 00 00 
  402ca6:	48 c7 84 24 68 02 00 	movq   $0x0,0x268(%rsp)
  402cad:	00 00 00 00 00 
  402cb2:	48 c7 84 24 60 02 00 	movq   $0x0,0x260(%rsp)
  402cb9:	00 00 00 00 00 
  402cbe:	48 c7 84 24 50 02 00 	movq   $0x0,0x250(%rsp)
  402cc5:	00 00 00 00 00 
  402cca:	48 c7 84 24 58 02 00 	movq   $0x0,0x258(%rsp)
  402cd1:	00 00 00 00 00 
  402cd6:	48 c7 84 24 78 02 00 	movq   $0x0,0x278(%rsp)
  402cdd:	00 00 00 00 00 
  402ce2:	48 c7 44 24 40 00 00 	movq   $0x0,0x40(%rsp)
  402ce9:	00 00 
  402ceb:	48 c7 44 24 48 00 00 	movq   $0x0,0x48(%rsp)
  402cf2:	00 00 
  402cf4:	48 c7 44 24 50 00 00 	movq   $0x0,0x50(%rsp)
  402cfb:	00 00 
  402cfd:	e8 5e ea ff ff       	callq  401760 <_ZNSsC1EPKcRKSaIcE@plt>
  402d02:	48 8d 74 24 20       	lea    0x20(%rsp),%rsi
  402d07:	48 8d bc 24 20 01 00 	lea    0x120(%rsp),%rdi
  402d0e:	00 
  402d0f:	ba 01 00 00 00       	mov    $0x1,%edx
  402d14:	e8 37 ea ff ff       	callq  401750 <_ZN2cv6imreadERKSsi@plt>
  402d19:	48 8b 84 24 38 01 00 	mov    0x138(%rsp),%rax
  402d20:	00 
  402d21:	48 85 c0             	test   %rax,%rax
  402d24:	74 04                	je     402d2a <_Z11decodeImagePfS_S_+0x3ba>
  402d26:	f0 83 00 01          	lock addl $0x1,(%rax)
  402d2a:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  402d2f:	48 85 c0             	test   %rax,%rax
  402d32:	74 0a                	je     402d3e <_Z11decodeImagePfS_S_+0x3ce>
  402d34:	f0 83 28 01          	lock subl $0x1,(%rax)
  402d38:	0f 84 d2 03 00 00    	je     403110 <_Z11decodeImagePfS_S_+0x7a0>
  402d3e:	48 8b 84 24 a0 00 00 	mov    0xa0(%rsp),%rax
  402d45:	00 
  402d46:	48 c7 84 24 90 00 00 	movq   $0x0,0x90(%rsp)
  402d4d:	00 00 00 00 00 
  402d52:	48 c7 84 24 88 00 00 	movq   $0x0,0x88(%rsp)
  402d59:	00 00 00 00 00 
  402d5e:	48 c7 84 24 80 00 00 	movq   $0x0,0x80(%rsp)
  402d65:	00 00 00 00 00 
  402d6a:	48 c7 44 24 70 00 00 	movq   $0x0,0x70(%rsp)
  402d71:	00 00 
  402d73:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402d79:	83 7c 24 64 02       	cmpl   $0x2,0x64(%rsp)
  402d7e:	8b 84 24 20 01 00 00 	mov    0x120(%rsp),%eax
  402d85:	48 c7 44 24 78 00 00 	movq   $0x0,0x78(%rsp)
  402d8c:	00 00 
  402d8e:	89 44 24 60          	mov    %eax,0x60(%rsp)
  402d92:	7f 10                	jg     402da4 <_Z11decodeImagePfS_S_+0x434>
  402d94:	8b 84 24 24 01 00 00 	mov    0x124(%rsp),%eax
  402d9b:	83 f8 02             	cmp    $0x2,%eax
  402d9e:	0f 8e 1c 03 00 00    	jle    4030c0 <_Z11decodeImagePfS_S_+0x750>
  402da4:	48 8d b4 24 20 01 00 	lea    0x120(%rsp),%rsi
  402dab:	00 
  402dac:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  402db1:	e8 4a e7 ff ff       	callq  401500 <_ZN2cv3Mat8copySizeERKS0_@plt>
  402db6:	c5 f9 6f 84 24 30 01 	vmovdqa 0x130(%rsp),%xmm0
  402dbd:	00 00 
  402dbf:	48 8b 84 24 38 01 00 	mov    0x138(%rsp),%rax
  402dc6:	00 
  402dc7:	c5 f9 7f 44 24 70    	vmovdqa %xmm0,0x70(%rsp)
  402dcd:	48 85 c0             	test   %rax,%rax
  402dd0:	c5 f9 6f 84 24 40 01 	vmovdqa 0x140(%rsp),%xmm0
  402dd7:	00 00 
  402dd9:	c5 f9 7f 84 24 80 00 	vmovdqa %xmm0,0x80(%rsp)
  402de0:	00 00 
  402de2:	c5 f9 6f 84 24 50 01 	vmovdqa 0x150(%rsp),%xmm0
  402de9:	00 00 
  402deb:	c5 f9 7f 84 24 90 00 	vmovdqa %xmm0,0x90(%rsp)
  402df2:	00 00 
  402df4:	74 0a                	je     402e00 <_Z11decodeImagePfS_S_+0x490>
  402df6:	f0 83 28 01          	lock subl $0x1,(%rax)
  402dfa:	0f 84 20 03 00 00    	je     403120 <_Z11decodeImagePfS_S_+0x7b0>
  402e00:	48 8b 84 24 60 01 00 	mov    0x160(%rsp),%rax
  402e07:	00 
  402e08:	48 8b bc 24 68 01 00 	mov    0x168(%rsp),%rdi
  402e0f:	00 
  402e10:	48 c7 84 24 50 01 00 	movq   $0x0,0x150(%rsp)
  402e17:	00 00 00 00 00 
  402e1c:	48 c7 84 24 48 01 00 	movq   $0x0,0x148(%rsp)
  402e23:	00 00 00 00 00 
  402e28:	48 c7 84 24 40 01 00 	movq   $0x0,0x140(%rsp)
  402e2f:	00 00 00 00 00 
  402e34:	48 c7 84 24 30 01 00 	movq   $0x0,0x130(%rsp)
  402e3b:	00 00 00 00 00 
  402e40:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402e46:	48 8d 84 24 70 01 00 	lea    0x170(%rsp),%rax
  402e4d:	00 
  402e4e:	48 c7 84 24 38 01 00 	movq   $0x0,0x138(%rsp)
  402e55:	00 00 00 00 00 
  402e5a:	48 39 c7             	cmp    %rax,%rdi
  402e5d:	74 05                	je     402e64 <_Z11decodeImagePfS_S_+0x4f4>
  402e5f:	e8 6c e8 ff ff       	callq  4016d0 <_ZN2cv8fastFreeEPv@plt>
  402e64:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  402e69:	48 8d 78 e8          	lea    -0x18(%rax),%rdi
  402e6d:	48 81 ff c0 51 60 00 	cmp    $0x6051c0,%rdi
  402e74:	0f 85 e5 02 00 00    	jne    40315f <_Z11decodeImagePfS_S_+0x7ef>
  402e7a:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  402e7f:	48 89 de             	mov    %rbx,%rsi
  402e82:	44 8b 74 24 68       	mov    0x68(%rsp),%r14d
  402e87:	44 8b 7c 24 6c       	mov    0x6c(%rsp),%r15d
  402e8c:	e8 af e8 ff ff       	callq  401740 <_ZN2cv5splitERKNS_3MatEPS0_@plt>
  402e91:	48 8b 94 24 a8 01 00 	mov    0x1a8(%rsp),%rdx
  402e98:	00 
  402e99:	48 8b b4 24 a0 01 00 	mov    0x1a0(%rsp),%rsi
  402ea0:	00 
  402ea1:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  402ea6:	c6 04 24 00          	movb   $0x0,(%rsp)
  402eaa:	e8 91 0e 00 00       	callq  403d40 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag>
  402eaf:	45 0f af f7          	imul   %r15d,%r14d
  402eb3:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  402eb8:	4c 89 ef             	mov    %r13,%rdi
  402ebb:	4d 63 f6             	movslq %r14d,%r14
  402ebe:	49 c1 e6 02          	shl    $0x2,%r14
  402ec2:	4c 89 f2             	mov    %r14,%rdx
  402ec5:	e8 26 e6 ff ff       	callq  4014f0 <memcpy@plt>
  402eca:	48 8b 94 24 08 02 00 	mov    0x208(%rsp),%rdx
  402ed1:	00 
  402ed2:	48 8b b4 24 00 02 00 	mov    0x200(%rsp),%rsi
  402ed9:	00 
  402eda:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  402edf:	c6 04 24 00          	movb   $0x0,(%rsp)
  402ee3:	e8 58 0e 00 00       	callq  403d40 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag>
  402ee8:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  402eed:	4c 89 f2             	mov    %r14,%rdx
  402ef0:	4c 89 e7             	mov    %r12,%rdi
  402ef3:	e8 f8 e5 ff ff       	callq  4014f0 <memcpy@plt>
  402ef8:	48 8b 94 24 68 02 00 	mov    0x268(%rsp),%rdx
  402eff:	00 
  402f00:	48 8b b4 24 60 02 00 	mov    0x260(%rsp),%rsi
  402f07:	00 
  402f08:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  402f0d:	c6 04 24 00          	movb   $0x0,(%rsp)
  402f11:	e8 2a 0e 00 00       	callq  403d40 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag>
  402f16:	4c 8b 64 24 40       	mov    0x40(%rsp),%r12
  402f1b:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  402f20:	4c 89 f2             	mov    %r14,%rdx
  402f23:	4c 89 e6             	mov    %r12,%rsi
  402f26:	e8 c5 e5 ff ff       	callq  4014f0 <memcpy@plt>
  402f2b:	4d 85 e4             	test   %r12,%r12
  402f2e:	74 08                	je     402f38 <_Z11decodeImagePfS_S_+0x5c8>
  402f30:	4c 89 e7             	mov    %r12,%rdi
  402f33:	e8 98 e6 ff ff       	callq  4015d0 <_ZdlPv@plt>
  402f38:	4c 8d a4 24 a0 02 00 	lea    0x2a0(%rsp),%r12
  402f3f:	00 
  402f40:	49 83 ec 60          	sub    $0x60,%r12
  402f44:	49 8b 44 24 18       	mov    0x18(%r12),%rax
  402f49:	48 85 c0             	test   %rax,%rax
  402f4c:	74 0a                	je     402f58 <_Z11decodeImagePfS_S_+0x5e8>
  402f4e:	f0 83 28 01          	lock subl $0x1,(%rax)
  402f52:	0f 84 58 01 00 00    	je     4030b0 <_Z11decodeImagePfS_S_+0x740>
  402f58:	49 8b 44 24 40       	mov    0x40(%r12),%rax
  402f5d:	49 c7 44 24 30 00 00 	movq   $0x0,0x30(%r12)
  402f64:	00 00 
  402f66:	49 c7 44 24 28 00 00 	movq   $0x0,0x28(%r12)
  402f6d:	00 00 
  402f6f:	49 c7 44 24 20 00 00 	movq   $0x0,0x20(%r12)
  402f76:	00 00 
  402f78:	49 c7 44 24 10 00 00 	movq   $0x0,0x10(%r12)
  402f7f:	00 00 
  402f81:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402f87:	49 8b 7c 24 48       	mov    0x48(%r12),%rdi
  402f8c:	49 8d 44 24 50       	lea    0x50(%r12),%rax
  402f91:	49 c7 44 24 18 00 00 	movq   $0x0,0x18(%r12)
  402f98:	00 00 
  402f9a:	48 39 c7             	cmp    %rax,%rdi
  402f9d:	74 05                	je     402fa4 <_Z11decodeImagePfS_S_+0x634>
  402f9f:	e8 2c e7 ff ff       	callq  4016d0 <_ZN2cv8fastFreeEPv@plt>
  402fa4:	49 39 dc             	cmp    %rbx,%r12
  402fa7:	75 97                	jne    402f40 <_Z11decodeImagePfS_S_+0x5d0>
  402fa9:	48 8b 84 24 d8 00 00 	mov    0xd8(%rsp),%rax
  402fb0:	00 
  402fb1:	48 85 c0             	test   %rax,%rax
  402fb4:	74 0a                	je     402fc0 <_Z11decodeImagePfS_S_+0x650>
  402fb6:	f0 83 28 01          	lock subl $0x1,(%rax)
  402fba:	0f 84 78 01 00 00    	je     403138 <_Z11decodeImagePfS_S_+0x7c8>
  402fc0:	48 8b 84 24 00 01 00 	mov    0x100(%rsp),%rax
  402fc7:	00 
  402fc8:	48 8b bc 24 08 01 00 	mov    0x108(%rsp),%rdi
  402fcf:	00 
  402fd0:	48 c7 84 24 f0 00 00 	movq   $0x0,0xf0(%rsp)
  402fd7:	00 00 00 00 00 
  402fdc:	48 c7 84 24 e8 00 00 	movq   $0x0,0xe8(%rsp)
  402fe3:	00 00 00 00 00 
  402fe8:	48 c7 84 24 e0 00 00 	movq   $0x0,0xe0(%rsp)
  402fef:	00 00 00 00 00 
  402ff4:	48 c7 84 24 d0 00 00 	movq   $0x0,0xd0(%rsp)
  402ffb:	00 00 00 00 00 
  403000:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  403006:	48 8d 84 24 10 01 00 	lea    0x110(%rsp),%rax
  40300d:	00 
  40300e:	48 c7 84 24 d8 00 00 	movq   $0x0,0xd8(%rsp)
  403015:	00 00 00 00 00 
  40301a:	48 39 c7             	cmp    %rax,%rdi
  40301d:	74 05                	je     403024 <_Z11decodeImagePfS_S_+0x6b4>
  40301f:	e8 ac e6 ff ff       	callq  4016d0 <_ZN2cv8fastFreeEPv@plt>
  403024:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  403029:	48 85 c0             	test   %rax,%rax
  40302c:	74 0a                	je     403038 <_Z11decodeImagePfS_S_+0x6c8>
  40302e:	f0 83 28 01          	lock subl $0x1,(%rax)
  403032:	0f 84 18 01 00 00    	je     403150 <_Z11decodeImagePfS_S_+0x7e0>
  403038:	48 8b 84 24 a0 00 00 	mov    0xa0(%rsp),%rax
  40303f:	00 
  403040:	48 8b bc 24 a8 00 00 	mov    0xa8(%rsp),%rdi
  403047:	00 
  403048:	48 c7 84 24 90 00 00 	movq   $0x0,0x90(%rsp)
  40304f:	00 00 00 00 00 
  403054:	48 c7 84 24 88 00 00 	movq   $0x0,0x88(%rsp)
  40305b:	00 00 00 00 00 
  403060:	48 c7 84 24 80 00 00 	movq   $0x0,0x80(%rsp)
  403067:	00 00 00 00 00 
  40306c:	48 c7 44 24 70 00 00 	movq   $0x0,0x70(%rsp)
  403073:	00 00 
  403075:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40307b:	48 8d 84 24 b0 00 00 	lea    0xb0(%rsp),%rax
  403082:	00 
  403083:	48 c7 44 24 78 00 00 	movq   $0x0,0x78(%rsp)
  40308a:	00 00 
  40308c:	48 39 c7             	cmp    %rax,%rdi
  40308f:	74 05                	je     403096 <_Z11decodeImagePfS_S_+0x726>
  403091:	e8 3a e6 ff ff       	callq  4016d0 <_ZN2cv8fastFreeEPv@plt>
  403096:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
  40309a:	31 c0                	xor    %eax,%eax
  40309c:	5b                   	pop    %rbx
  40309d:	41 5c                	pop    %r12
  40309f:	41 5d                	pop    %r13
  4030a1:	41 5e                	pop    %r14
  4030a3:	41 5f                	pop    %r15
  4030a5:	5d                   	pop    %rbp
  4030a6:	c3                   	retq   
  4030a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4030ae:	00 00 
  4030b0:	4c 89 e7             	mov    %r12,%rdi
  4030b3:	e8 08 e5 ff ff       	callq  4015c0 <_ZN2cv3Mat10deallocateEv@plt>
  4030b8:	e9 9b fe ff ff       	jmpq   402f58 <_Z11decodeImagePfS_S_+0x5e8>
  4030bd:	0f 1f 00             	nopl   (%rax)
  4030c0:	89 44 24 64          	mov    %eax,0x64(%rsp)
  4030c4:	8b 84 24 28 01 00 00 	mov    0x128(%rsp),%eax
  4030cb:	89 44 24 68          	mov    %eax,0x68(%rsp)
  4030cf:	8b 84 24 2c 01 00 00 	mov    0x12c(%rsp),%eax
  4030d6:	89 44 24 6c          	mov    %eax,0x6c(%rsp)
  4030da:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  4030e1:	00 
  4030e2:	48 8b 10             	mov    (%rax),%rdx
  4030e5:	48 8b 84 24 a8 00 00 	mov    0xa8(%rsp),%rax
  4030ec:	00 
  4030ed:	48 89 10             	mov    %rdx,(%rax)
  4030f0:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  4030f7:	00 
  4030f8:	48 8b 50 08          	mov    0x8(%rax),%rdx
  4030fc:	48 8b 84 24 a8 00 00 	mov    0xa8(%rsp),%rax
  403103:	00 
  403104:	48 89 50 08          	mov    %rdx,0x8(%rax)
  403108:	e9 a9 fc ff ff       	jmpq   402db6 <_Z11decodeImagePfS_S_+0x446>
  40310d:	0f 1f 00             	nopl   (%rax)
  403110:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  403115:	e8 a6 e4 ff ff       	callq  4015c0 <_ZN2cv3Mat10deallocateEv@plt>
  40311a:	e9 1f fc ff ff       	jmpq   402d3e <_Z11decodeImagePfS_S_+0x3ce>
  40311f:	90                   	nop
  403120:	48 8d bc 24 20 01 00 	lea    0x120(%rsp),%rdi
  403127:	00 
  403128:	e8 93 e4 ff ff       	callq  4015c0 <_ZN2cv3Mat10deallocateEv@plt>
  40312d:	e9 ce fc ff ff       	jmpq   402e00 <_Z11decodeImagePfS_S_+0x490>
  403132:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  403138:	48 8d bc 24 c0 00 00 	lea    0xc0(%rsp),%rdi
  40313f:	00 
  403140:	e8 7b e4 ff ff       	callq  4015c0 <_ZN2cv3Mat10deallocateEv@plt>
  403145:	e9 76 fe ff ff       	jmpq   402fc0 <_Z11decodeImagePfS_S_+0x650>
  40314a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  403150:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  403155:	e8 66 e4 ff ff       	callq  4015c0 <_ZN2cv3Mat10deallocateEv@plt>
  40315a:	e9 d9 fe ff ff       	jmpq   403038 <_Z11decodeImagePfS_S_+0x6c8>
  40315f:	b9 e0 16 40 00       	mov    $0x4016e0,%ecx
  403164:	48 8d 57 10          	lea    0x10(%rdi),%rdx
  403168:	48 85 c9             	test   %rcx,%rcx
  40316b:	74 46                	je     4031b3 <_Z11decodeImagePfS_S_+0x843>
  40316d:	83 c8 ff             	or     $0xffffffff,%eax
  403170:	f0 0f c1 02          	lock xadd %eax,(%rdx)
  403174:	85 c0                	test   %eax,%eax
  403176:	0f 8f fe fc ff ff    	jg     402e7a <_Z11decodeImagePfS_S_+0x50a>
  40317c:	48 8d b4 24 20 01 00 	lea    0x120(%rsp),%rsi
  403183:	00 
  403184:	e8 87 e5 ff ff       	callq  401710 <_ZNSs4_Rep10_M_destroyERKSaIcE@plt>
  403189:	e9 ec fc ff ff       	jmpq   402e7a <_Z11decodeImagePfS_S_+0x50a>
  40318e:	48 89 c3             	mov    %rax,%rbx
  403191:	c5 f8 77             	vzeroupper 
  403194:	48 8d bc 24 c0 00 00 	lea    0xc0(%rsp),%rdi
  40319b:	00 
  40319c:	e8 8f 00 00 00       	callq  403230 <_ZN2cv3MatD1Ev>
  4031a1:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  4031a6:	e8 85 00 00 00       	callq  403230 <_ZN2cv3MatD1Ev>
  4031ab:	48 89 df             	mov    %rbx,%rdi
  4031ae:	e8 6d e4 ff ff       	callq  401620 <_Unwind_Resume@plt>
  4031b3:	8b 50 f8             	mov    -0x8(%rax),%edx
  4031b6:	8d 4a ff             	lea    -0x1(%rdx),%ecx
  4031b9:	89 48 f8             	mov    %ecx,-0x8(%rax)
  4031bc:	89 d0                	mov    %edx,%eax
  4031be:	eb b4                	jmp    403174 <_Z11decodeImagePfS_S_+0x804>
  4031c0:	49 89 c4             	mov    %rax,%r12
  4031c3:	c5 f8 77             	vzeroupper 
  4031c6:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  4031cb:	48 8d 74 24 1f       	lea    0x1f(%rsp),%rsi
  4031d0:	48 8d 78 e8          	lea    -0x18(%rax),%rdi
  4031d4:	e8 a7 e4 ff ff       	callq  401680 <_ZNSs4_Rep10_M_disposeERKSaIcE@plt>
  4031d9:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  4031de:	48 85 ff             	test   %rdi,%rdi
  4031e1:	74 05                	je     4031e8 <_Z11decodeImagePfS_S_+0x878>
  4031e3:	e8 e8 e3 ff ff       	callq  4015d0 <_ZdlPv@plt>
  4031e8:	48 8d bb c0 00 00 00 	lea    0xc0(%rbx),%rdi
  4031ef:	e8 3c 00 00 00       	callq  403230 <_ZN2cv3MatD1Ev>
  4031f4:	48 8d 7b 60          	lea    0x60(%rbx),%rdi
  4031f8:	e8 33 00 00 00       	callq  403230 <_ZN2cv3MatD1Ev>
  4031fd:	48 89 df             	mov    %rbx,%rdi
  403200:	4c 89 e3             	mov    %r12,%rbx
  403203:	e8 28 00 00 00       	callq  403230 <_ZN2cv3MatD1Ev>
  403208:	eb 8a                	jmp    403194 <_Z11decodeImagePfS_S_+0x824>
  40320a:	49 89 c4             	mov    %rax,%r12
  40320d:	c5 f8 77             	vzeroupper 
  403210:	eb c7                	jmp    4031d9 <_Z11decodeImagePfS_S_+0x869>
  403212:	48 89 c3             	mov    %rax,%rbx
  403215:	c5 f8 77             	vzeroupper 
  403218:	eb 87                	jmp    4031a1 <_Z11decodeImagePfS_S_+0x831>
  40321a:	48 8d bc 24 20 01 00 	lea    0x120(%rsp),%rdi
  403221:	00 
  403222:	49 89 c4             	mov    %rax,%r12
  403225:	c5 f8 77             	vzeroupper 
  403228:	e8 03 00 00 00       	callq  403230 <_ZN2cv3MatD1Ev>
  40322d:	eb 97                	jmp    4031c6 <_Z11decodeImagePfS_S_+0x856>
  40322f:	90                   	nop

0000000000403230 <_ZN2cv3MatD1Ev>:
  403230:	53                   	push   %rbx
  403231:	48 8b 47 18          	mov    0x18(%rdi),%rax
  403235:	48 89 fb             	mov    %rdi,%rbx
  403238:	48 85 c0             	test   %rax,%rax
  40323b:	74 06                	je     403243 <_ZN2cv3MatD1Ev+0x13>
  40323d:	f0 83 28 01          	lock subl $0x1,(%rax)
  403241:	74 55                	je     403298 <_ZN2cv3MatD1Ev+0x68>
  403243:	48 8b 43 40          	mov    0x40(%rbx),%rax
  403247:	48 8b 7b 48          	mov    0x48(%rbx),%rdi
  40324b:	48 83 c3 50          	add    $0x50,%rbx
  40324f:	48 c7 43 e0 00 00 00 	movq   $0x0,-0x20(%rbx)
  403256:	00 
  403257:	48 c7 43 d8 00 00 00 	movq   $0x0,-0x28(%rbx)
  40325e:	00 
  40325f:	48 c7 43 d0 00 00 00 	movq   $0x0,-0x30(%rbx)
  403266:	00 
  403267:	48 c7 43 c0 00 00 00 	movq   $0x0,-0x40(%rbx)
  40326e:	00 
  40326f:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  403275:	48 c7 43 c8 00 00 00 	movq   $0x0,-0x38(%rbx)
  40327c:	00 
  40327d:	48 39 df             	cmp    %rbx,%rdi
  403280:	74 0e                	je     403290 <_ZN2cv3MatD1Ev+0x60>
  403282:	5b                   	pop    %rbx
  403283:	e9 48 e4 ff ff       	jmpq   4016d0 <_ZN2cv8fastFreeEPv@plt>
  403288:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40328f:	00 
  403290:	5b                   	pop    %rbx
  403291:	c3                   	retq   
  403292:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  403298:	e8 23 e3 ff ff       	callq  4015c0 <_ZN2cv3Mat10deallocateEv@plt>
  40329d:	0f 1f 00             	nopl   (%rax)
  4032a0:	eb a1                	jmp    403243 <_ZN2cv3MatD1Ev+0x13>
  4032a2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4032a9:	00 00 00 
  4032ac:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004032b0 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev>:
  4032b0:	41 54                	push   %r12
  4032b2:	49 89 fc             	mov    %rdi,%r12
  4032b5:	55                   	push   %rbp
  4032b6:	53                   	push   %rbx
  4032b7:	48 8b 6f 08          	mov    0x8(%rdi),%rbp
  4032bb:	48 8b 1f             	mov    (%rdi),%rbx
  4032be:	48 39 dd             	cmp    %rbx,%rbp
  4032c1:	75 5a                	jne    40331d <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x6d>
  4032c3:	e9 95 00 00 00       	jmpq   40335d <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0xad>
  4032c8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4032cf:	00 
  4032d0:	48 8b 43 40          	mov    0x40(%rbx),%rax
  4032d4:	48 c7 43 30 00 00 00 	movq   $0x0,0x30(%rbx)
  4032db:	00 
  4032dc:	48 c7 43 28 00 00 00 	movq   $0x0,0x28(%rbx)
  4032e3:	00 
  4032e4:	48 c7 43 20 00 00 00 	movq   $0x0,0x20(%rbx)
  4032eb:	00 
  4032ec:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
  4032f3:	00 
  4032f4:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4032fa:	48 8b 7b 48          	mov    0x48(%rbx),%rdi
  4032fe:	48 8d 43 50          	lea    0x50(%rbx),%rax
  403302:	48 c7 43 18 00 00 00 	movq   $0x0,0x18(%rbx)
  403309:	00 
  40330a:	48 39 c7             	cmp    %rax,%rdi
  40330d:	74 05                	je     403314 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x64>
  40330f:	e8 bc e3 ff ff       	callq  4016d0 <_ZN2cv8fastFreeEPv@plt>
  403314:	48 83 c3 60          	add    $0x60,%rbx
  403318:	48 39 dd             	cmp    %rbx,%rbp
  40331b:	74 23                	je     403340 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x90>
  40331d:	48 8b 43 18          	mov    0x18(%rbx),%rax
  403321:	48 85 c0             	test   %rax,%rax
  403324:	74 aa                	je     4032d0 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x20>
  403326:	f0 83 28 01          	lock subl $0x1,(%rax)
  40332a:	75 a4                	jne    4032d0 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x20>
  40332c:	48 89 df             	mov    %rbx,%rdi
  40332f:	e8 8c e2 ff ff       	callq  4015c0 <_ZN2cv3Mat10deallocateEv@plt>
  403334:	eb 9a                	jmp    4032d0 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x20>
  403336:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40333d:	00 00 00 
  403340:	49 8b 3c 24          	mov    (%r12),%rdi
  403344:	48 85 ff             	test   %rdi,%rdi
  403347:	74 0f                	je     403358 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0xa8>
  403349:	5b                   	pop    %rbx
  40334a:	5d                   	pop    %rbp
  40334b:	41 5c                	pop    %r12
  40334d:	e9 7e e2 ff ff       	jmpq   4015d0 <_ZdlPv@plt>
  403352:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  403358:	5b                   	pop    %rbx
  403359:	5d                   	pop    %rbp
  40335a:	41 5c                	pop    %r12
  40335c:	c3                   	retq   
  40335d:	48 89 ef             	mov    %rbp,%rdi
  403360:	eb e2                	jmp    403344 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x94>
  403362:	49 8b 3c 24          	mov    (%r12),%rdi
  403366:	48 89 c3             	mov    %rax,%rbx
  403369:	48 85 ff             	test   %rdi,%rdi
  40336c:	74 10                	je     40337e <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0xce>
  40336e:	c5 f8 77             	vzeroupper 
  403371:	e8 5a e2 ff ff       	callq  4015d0 <_ZdlPv@plt>
  403376:	48 89 df             	mov    %rbx,%rdi
  403379:	e8 a2 e2 ff ff       	callq  401620 <_Unwind_Resume@plt>
  40337e:	c5 f8 77             	vzeroupper 
  403381:	eb f3                	jmp    403376 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0xc6>
  403383:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40338a:	00 00 00 
  40338d:	0f 1f 00             	nopl   (%rax)

0000000000403390 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_>:
  403390:	55                   	push   %rbp
  403391:	48 89 e5             	mov    %rsp,%rbp
  403394:	41 57                	push   %r15
  403396:	49 89 ff             	mov    %rdi,%r15
  403399:	41 56                	push   %r14
  40339b:	49 89 f6             	mov    %rsi,%r14
  40339e:	41 55                	push   %r13
  4033a0:	41 54                	push   %r12
  4033a2:	53                   	push   %rbx
  4033a3:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4033a7:	48 81 ec 90 00 00 00 	sub    $0x90,%rsp
  4033ae:	48 8b 7f 08          	mov    0x8(%rdi),%rdi
  4033b2:	49 3b 7f 10          	cmp    0x10(%r15),%rdi
  4033b6:	0f 84 3b 04 00 00    	je     4037f7 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x467>
  4033bc:	31 c0                	xor    %eax,%eax
  4033be:	48 85 ff             	test   %rdi,%rdi
  4033c1:	0f 84 8c 00 00 00    	je     403453 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0xc3>
  4033c7:	48 8b 4f c0          	mov    -0x40(%rdi),%rcx
  4033cb:	48 8b 47 b0          	mov    -0x50(%rdi),%rax
  4033cf:	c5 fa 6f 47 a0       	vmovdqu -0x60(%rdi),%xmm0
  4033d4:	48 89 4f 20          	mov    %rcx,0x20(%rdi)
  4033d8:	48 8b 4f c8          	mov    -0x38(%rdi),%rcx
  4033dc:	48 89 47 10          	mov    %rax,0x10(%rdi)
  4033e0:	48 8b 47 b8          	mov    -0x48(%rdi),%rax
  4033e4:	c5 fa 7f 07          	vmovdqu %xmm0,(%rdi)
  4033e8:	48 89 4f 28          	mov    %rcx,0x28(%rdi)
  4033ec:	48 8b 4f d0          	mov    -0x30(%rdi),%rcx
  4033f0:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  4033f4:	48 85 c0             	test   %rax,%rax
  4033f7:	48 89 4f 30          	mov    %rcx,0x30(%rdi)
  4033fb:	48 8b 4f d8          	mov    -0x28(%rdi),%rcx
  4033ff:	48 89 47 18          	mov    %rax,0x18(%rdi)
  403403:	c5 fa 7f 47 50       	vmovdqu %xmm0,0x50(%rdi)
  403408:	48 89 4f 38          	mov    %rcx,0x38(%rdi)
  40340c:	48 8d 4f 08          	lea    0x8(%rdi),%rcx
  403410:	48 89 4f 40          	mov    %rcx,0x40(%rdi)
  403414:	48 8d 4f 50          	lea    0x50(%rdi),%rcx
  403418:	48 89 4f 48          	mov    %rcx,0x48(%rdi)
  40341c:	8b 4f a4             	mov    -0x5c(%rdi),%ecx
  40341f:	74 07                	je     403428 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x98>
  403421:	f0 83 00 01          	lock addl $0x1,(%rax)
  403425:	8b 4f a4             	mov    -0x5c(%rdi),%ecx
  403428:	83 f9 02             	cmp    $0x2,%ecx
  40342b:	0f 8f 57 03 00 00    	jg     403788 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x3f8>
  403431:	48 8b 4f e8          	mov    -0x18(%rdi),%rcx
  403435:	48 8b 47 48          	mov    0x48(%rdi),%rax
  403439:	48 8b 09             	mov    (%rcx),%rcx
  40343c:	48 89 08             	mov    %rcx,(%rax)
  40343f:	48 8b 4f e8          	mov    -0x18(%rdi),%rcx
  403443:	48 8b 47 48          	mov    0x48(%rdi),%rax
  403447:	48 8b 49 08          	mov    0x8(%rcx),%rcx
  40344b:	48 89 48 08          	mov    %rcx,0x8(%rax)
  40344f:	49 8b 47 08          	mov    0x8(%r15),%rax
  403453:	48 8b 4a 20          	mov    0x20(%rdx),%rcx
  403457:	48 83 c0 60          	add    $0x60,%rax
  40345b:	4c 8d 6c 24 30       	lea    0x30(%rsp),%r13
  403460:	49 89 47 08          	mov    %rax,0x8(%r15)
  403464:	48 8b 42 10          	mov    0x10(%rdx),%rax
  403468:	c5 fa 6f 02          	vmovdqu (%rdx),%xmm0
  40346c:	48 89 4c 24 50       	mov    %rcx,0x50(%rsp)
  403471:	48 8b 4a 28          	mov    0x28(%rdx),%rcx
  403475:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  40347a:	48 8b 42 18          	mov    0x18(%rdx),%rax
  40347e:	c5 f9 7f 44 24 30    	vmovdqa %xmm0,0x30(%rsp)
  403484:	48 89 4c 24 58       	mov    %rcx,0x58(%rsp)
  403489:	48 8b 4a 30          	mov    0x30(%rdx),%rcx
  40348d:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  403491:	48 85 c0             	test   %rax,%rax
  403494:	48 89 4c 24 60       	mov    %rcx,0x60(%rsp)
  403499:	48 8b 4a 38          	mov    0x38(%rdx),%rcx
  40349d:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  4034a2:	c5 f9 7f 84 24 80 00 	vmovdqa %xmm0,0x80(%rsp)
  4034a9:	00 00 
  4034ab:	48 89 4c 24 68       	mov    %rcx,0x68(%rsp)
  4034b0:	49 8d 4d 08          	lea    0x8(%r13),%rcx
  4034b4:	48 89 4c 24 70       	mov    %rcx,0x70(%rsp)
  4034b9:	49 8d 4d 50          	lea    0x50(%r13),%rcx
  4034bd:	48 89 4c 24 78       	mov    %rcx,0x78(%rsp)
  4034c2:	8b 4a 04             	mov    0x4(%rdx),%ecx
  4034c5:	74 07                	je     4034ce <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x13e>
  4034c7:	f0 83 00 01          	lock addl $0x1,(%rax)
  4034cb:	8b 4a 04             	mov    0x4(%rdx),%ecx
  4034ce:	83 f9 02             	cmp    $0x2,%ecx
  4034d1:	0f 8f d9 02 00 00    	jg     4037b0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x420>
  4034d7:	48 8b 42 48          	mov    0x48(%rdx),%rax
  4034db:	48 8b 08             	mov    (%rax),%rcx
  4034de:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  4034e3:	48 89 08             	mov    %rcx,(%rax)
  4034e6:	48 8b 42 48          	mov    0x48(%rdx),%rax
  4034ea:	48 8b 50 08          	mov    0x8(%rax),%rdx
  4034ee:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  4034f3:	48 89 50 08          	mov    %rdx,0x8(%rax)
  4034f7:	4d 8b 67 08          	mov    0x8(%r15),%r12
  4034fb:	48 b8 ab aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaab,%rax
  403502:	aa aa aa 
  403505:	49 8d 5c 24 a0       	lea    -0x60(%r12),%rbx
  40350a:	49 81 ec c0 00 00 00 	sub    $0xc0,%r12
  403511:	4d 29 f4             	sub    %r14,%r12
  403514:	49 c1 fc 05          	sar    $0x5,%r12
  403518:	4c 0f af e0          	imul   %rax,%r12
  40351c:	4d 85 e4             	test   %r12,%r12
  40351f:	7f 76                	jg     403597 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x207>
  403521:	e9 0a 01 00 00       	jmpq   403630 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2a0>
  403526:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40352d:	00 00 00 
  403530:	89 43 04             	mov    %eax,0x4(%rbx)
  403533:	8b 43 a8             	mov    -0x58(%rbx),%eax
  403536:	48 8b 53 e8          	mov    -0x18(%rbx),%rdx
  40353a:	89 43 08             	mov    %eax,0x8(%rbx)
  40353d:	8b 43 ac             	mov    -0x54(%rbx),%eax
  403540:	89 43 0c             	mov    %eax,0xc(%rbx)
  403543:	48 8b 43 48          	mov    0x48(%rbx),%rax
  403547:	48 8b 12             	mov    (%rdx),%rdx
  40354a:	48 89 10             	mov    %rdx,(%rax)
  40354d:	48 8b 53 e8          	mov    -0x18(%rbx),%rdx
  403551:	48 8b 43 48          	mov    0x48(%rbx),%rax
  403555:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  403559:	48 89 50 08          	mov    %rdx,0x8(%rax)
  40355d:	48 8b 43 b0          	mov    -0x50(%rbx),%rax
  403561:	48 89 43 10          	mov    %rax,0x10(%rbx)
  403565:	48 8b 43 c0          	mov    -0x40(%rbx),%rax
  403569:	48 89 43 20          	mov    %rax,0x20(%rbx)
  40356d:	48 8b 43 c8          	mov    -0x38(%rbx),%rax
  403571:	48 89 43 28          	mov    %rax,0x28(%rbx)
  403575:	48 8b 43 d0          	mov    -0x30(%rbx),%rax
  403579:	48 89 43 30          	mov    %rax,0x30(%rbx)
  40357d:	48 8b 43 b8          	mov    -0x48(%rbx),%rax
  403581:	48 89 43 18          	mov    %rax,0x18(%rbx)
  403585:	48 8b 43 d8          	mov    -0x28(%rbx),%rax
  403589:	48 89 43 38          	mov    %rax,0x38(%rbx)
  40358d:	49 83 ec 01          	sub    $0x1,%r12
  403591:	0f 84 99 00 00 00    	je     403630 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2a0>
  403597:	4c 8d bb 40 ff ff ff 	lea    -0xc0(%rbx),%r15
  40359e:	48 83 eb 60          	sub    $0x60,%rbx
  4035a2:	4c 39 fb             	cmp    %r15,%rbx
  4035a5:	74 e6                	je     40358d <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x1fd>
  4035a7:	48 8b 43 b8          	mov    -0x48(%rbx),%rax
  4035ab:	48 85 c0             	test   %rax,%rax
  4035ae:	74 04                	je     4035b4 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x224>
  4035b0:	f0 83 00 01          	lock addl $0x1,(%rax)
  4035b4:	48 8b 43 18          	mov    0x18(%rbx),%rax
  4035b8:	48 85 c0             	test   %rax,%rax
  4035bb:	74 06                	je     4035c3 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x233>
  4035bd:	f0 83 28 01          	lock subl $0x1,(%rax)
  4035c1:	74 5d                	je     403620 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x290>
  4035c3:	48 8b 43 40          	mov    0x40(%rbx),%rax
  4035c7:	48 c7 43 30 00 00 00 	movq   $0x0,0x30(%rbx)
  4035ce:	00 
  4035cf:	48 c7 43 28 00 00 00 	movq   $0x0,0x28(%rbx)
  4035d6:	00 
  4035d7:	48 c7 43 20 00 00 00 	movq   $0x0,0x20(%rbx)
  4035de:	00 
  4035df:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
  4035e6:	00 
  4035e7:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4035ed:	83 7b 04 02          	cmpl   $0x2,0x4(%rbx)
  4035f1:	8b 43 a0             	mov    -0x60(%rbx),%eax
  4035f4:	48 c7 43 18 00 00 00 	movq   $0x0,0x18(%rbx)
  4035fb:	00 
  4035fc:	89 03                	mov    %eax,(%rbx)
  4035fe:	7f 0c                	jg     40360c <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x27c>
  403600:	8b 43 a4             	mov    -0x5c(%rbx),%eax
  403603:	83 f8 02             	cmp    $0x2,%eax
  403606:	0f 8e 24 ff ff ff    	jle    403530 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x1a0>
  40360c:	4c 89 fe             	mov    %r15,%rsi
  40360f:	48 89 df             	mov    %rbx,%rdi
  403612:	e8 e9 de ff ff       	callq  401500 <_ZN2cv3Mat8copySizeERKS0_@plt>
  403617:	e9 41 ff ff ff       	jmpq   40355d <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x1cd>
  40361c:	0f 1f 40 00          	nopl   0x0(%rax)
  403620:	48 89 df             	mov    %rbx,%rdi
  403623:	e8 98 df ff ff       	callq  4015c0 <_ZN2cv3Mat10deallocateEv@plt>
  403628:	eb 99                	jmp    4035c3 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x233>
  40362a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  403630:	4d 39 ee             	cmp    %r13,%r14
  403633:	0f 84 b4 01 00 00    	je     4037ed <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x45d>
  403639:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  40363e:	48 85 c0             	test   %rax,%rax
  403641:	74 04                	je     403647 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2b7>
  403643:	f0 83 00 01          	lock addl $0x1,(%rax)
  403647:	49 8b 46 18          	mov    0x18(%r14),%rax
  40364b:	48 85 c0             	test   %rax,%rax
  40364e:	74 0a                	je     40365a <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2ca>
  403650:	f0 83 28 01          	lock subl $0x1,(%rax)
  403654:	0f 84 86 01 00 00    	je     4037e0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x450>
  40365a:	49 8b 46 40          	mov    0x40(%r14),%rax
  40365e:	49 c7 46 30 00 00 00 	movq   $0x0,0x30(%r14)
  403665:	00 
  403666:	49 c7 46 28 00 00 00 	movq   $0x0,0x28(%r14)
  40366d:	00 
  40366e:	49 c7 46 20 00 00 00 	movq   $0x0,0x20(%r14)
  403675:	00 
  403676:	49 c7 46 10 00 00 00 	movq   $0x0,0x10(%r14)
  40367d:	00 
  40367e:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  403684:	41 83 7e 04 02       	cmpl   $0x2,0x4(%r14)
  403689:	8b 44 24 30          	mov    0x30(%rsp),%eax
  40368d:	49 c7 46 18 00 00 00 	movq   $0x0,0x18(%r14)
  403694:	00 
  403695:	41 89 06             	mov    %eax,(%r14)
  403698:	7f 0d                	jg     4036a7 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x317>
  40369a:	8b 44 24 34          	mov    0x34(%rsp),%eax
  40369e:	83 f8 02             	cmp    $0x2,%eax
  4036a1:	0f 8e a1 00 00 00    	jle    403748 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x3b8>
  4036a7:	4c 89 ee             	mov    %r13,%rsi
  4036aa:	4c 89 f7             	mov    %r14,%rdi
  4036ad:	e8 4e de ff ff       	callq  401500 <_ZN2cv3Mat8copySizeERKS0_@plt>
  4036b2:	c5 f9 6f 44 24 40    	vmovdqa 0x40(%rsp),%xmm0
  4036b8:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  4036bd:	c4 c1 7a 7f 46 10    	vmovdqu %xmm0,0x10(%r14)
  4036c3:	c5 f9 6f 44 24 50    	vmovdqa 0x50(%rsp),%xmm0
  4036c9:	c4 c1 7a 7f 46 20    	vmovdqu %xmm0,0x20(%r14)
  4036cf:	c5 f9 6f 44 24 60    	vmovdqa 0x60(%rsp),%xmm0
  4036d5:	c4 c1 7a 7f 46 30    	vmovdqu %xmm0,0x30(%r14)
  4036db:	48 85 c0             	test   %rax,%rax
  4036de:	74 0a                	je     4036ea <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x35a>
  4036e0:	f0 83 28 01          	lock subl $0x1,(%rax)
  4036e4:	0f 84 e6 00 00 00    	je     4037d0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x440>
  4036ea:	48 8b 7c 24 78       	mov    0x78(%rsp),%rdi
  4036ef:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
  4036f4:	49 83 c5 50          	add    $0x50,%r13
  4036f8:	48 c7 44 24 60 00 00 	movq   $0x0,0x60(%rsp)
  4036ff:	00 00 
  403701:	48 c7 44 24 58 00 00 	movq   $0x0,0x58(%rsp)
  403708:	00 00 
  40370a:	48 c7 44 24 50 00 00 	movq   $0x0,0x50(%rsp)
  403711:	00 00 
  403713:	48 c7 44 24 40 00 00 	movq   $0x0,0x40(%rsp)
  40371a:	00 00 
  40371c:	4c 39 ef             	cmp    %r13,%rdi
  40371f:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  403725:	48 c7 44 24 48 00 00 	movq   $0x0,0x48(%rsp)
  40372c:	00 00 
  40372e:	74 05                	je     403735 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x3a5>
  403730:	e8 9b df ff ff       	callq  4016d0 <_ZN2cv8fastFreeEPv@plt>
  403735:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
  403739:	5b                   	pop    %rbx
  40373a:	41 5c                	pop    %r12
  40373c:	41 5d                	pop    %r13
  40373e:	41 5e                	pop    %r14
  403740:	41 5f                	pop    %r15
  403742:	5d                   	pop    %rbp
  403743:	c3                   	retq   
  403744:	0f 1f 40 00          	nopl   0x0(%rax)
  403748:	41 89 46 04          	mov    %eax,0x4(%r14)
  40374c:	8b 44 24 38          	mov    0x38(%rsp),%eax
  403750:	48 8b 54 24 78       	mov    0x78(%rsp),%rdx
  403755:	41 89 46 08          	mov    %eax,0x8(%r14)
  403759:	8b 44 24 3c          	mov    0x3c(%rsp),%eax
  40375d:	48 8b 12             	mov    (%rdx),%rdx
  403760:	41 89 46 0c          	mov    %eax,0xc(%r14)
  403764:	49 8b 46 48          	mov    0x48(%r14),%rax
  403768:	48 89 10             	mov    %rdx,(%rax)
  40376b:	48 8b 54 24 78       	mov    0x78(%rsp),%rdx
  403770:	49 8b 46 48          	mov    0x48(%r14),%rax
  403774:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  403778:	48 89 50 08          	mov    %rdx,0x8(%rax)
  40377c:	e9 31 ff ff ff       	jmpq   4036b2 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x322>
  403781:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  403788:	48 8d 77 a0          	lea    -0x60(%rdi),%rsi
  40378c:	c7 47 04 00 00 00 00 	movl   $0x0,0x4(%rdi)
  403793:	48 89 54 24 28       	mov    %rdx,0x28(%rsp)
  403798:	e8 63 dd ff ff       	callq  401500 <_ZN2cv3Mat8copySizeERKS0_@plt>
  40379d:	49 8b 47 08          	mov    0x8(%r15),%rax
  4037a1:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
  4037a6:	e9 a8 fc ff ff       	jmpq   403453 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0xc3>
  4037ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4037b0:	48 89 d6             	mov    %rdx,%rsi
  4037b3:	4c 89 ef             	mov    %r13,%rdi
  4037b6:	c7 44 24 34 00 00 00 	movl   $0x0,0x34(%rsp)
  4037bd:	00 
  4037be:	e8 3d dd ff ff       	callq  401500 <_ZN2cv3Mat8copySizeERKS0_@plt>
  4037c3:	e9 2f fd ff ff       	jmpq   4034f7 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x167>
  4037c8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4037cf:	00 
  4037d0:	4c 89 ef             	mov    %r13,%rdi
  4037d3:	e8 e8 dd ff ff       	callq  4015c0 <_ZN2cv3Mat10deallocateEv@plt>
  4037d8:	e9 0d ff ff ff       	jmpq   4036ea <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x35a>
  4037dd:	0f 1f 00             	nopl   (%rax)
  4037e0:	4c 89 f7             	mov    %r14,%rdi
  4037e3:	e8 d8 dd ff ff       	callq  4015c0 <_ZN2cv3Mat10deallocateEv@plt>
  4037e8:	e9 6d fe ff ff       	jmpq   40365a <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2ca>
  4037ed:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  4037f2:	e9 e4 fe ff ff       	jmpq   4036db <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x34b>
  4037f7:	49 8b 07             	mov    (%r15),%rax
  4037fa:	48 b9 ab aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaab,%rcx
  403801:	aa aa aa 
  403804:	48 29 c7             	sub    %rax,%rdi
  403807:	48 c1 ff 05          	sar    $0x5,%rdi
  40380b:	48 0f af f9          	imul   %rcx,%rdi
  40380f:	48 85 ff             	test   %rdi,%rdi
  403812:	0f 85 be 01 00 00    	jne    4039d6 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x646>
  403818:	48 89 f3             	mov    %rsi,%rbx
  40381b:	be 01 00 00 00       	mov    $0x1,%esi
  403820:	48 29 c3             	sub    %rax,%rbx
  403823:	48 c1 fb 05          	sar    $0x5,%rbx
  403827:	48 0f af d9          	imul   %rcx,%rbx
  40382b:	48 8d 3c 76          	lea    (%rsi,%rsi,2),%rdi
  40382f:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  403834:	48 c1 e7 05          	shl    $0x5,%rdi
  403838:	48 89 54 24 28       	mov    %rdx,0x28(%rsp)
  40383d:	e8 de dc ff ff       	callq  401520 <_Znwm@plt>
  403842:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
  403847:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  40384c:	48 8d 04 5b          	lea    (%rbx,%rbx,2),%rax
  403850:	48 c1 e0 05          	shl    $0x5,%rax
  403854:	48 03 44 24 18       	add    0x18(%rsp),%rax
  403859:	48 89 c6             	mov    %rax,%rsi
  40385c:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  403861:	0f 84 87 00 00 00    	je     4038ee <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x55e>
  403867:	c5 fa 6f 02          	vmovdqu (%rdx),%xmm0
  40386b:	c5 fa 7f 00          	vmovdqu %xmm0,(%rax)
  40386f:	48 8b 42 10          	mov    0x10(%rdx),%rax
  403873:	48 8b 4a 20          	mov    0x20(%rdx),%rcx
  403877:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  40387b:	48 89 46 10          	mov    %rax,0x10(%rsi)
  40387f:	48 8b 42 18          	mov    0x18(%rdx),%rax
  403883:	48 89 4e 20          	mov    %rcx,0x20(%rsi)
  403887:	48 8b 4a 28          	mov    0x28(%rdx),%rcx
  40388b:	48 85 c0             	test   %rax,%rax
  40388e:	48 89 46 18          	mov    %rax,0x18(%rsi)
  403892:	48 89 4e 28          	mov    %rcx,0x28(%rsi)
  403896:	48 8b 4a 30          	mov    0x30(%rdx),%rcx
  40389a:	48 89 4e 30          	mov    %rcx,0x30(%rsi)
  40389e:	48 8b 4a 38          	mov    0x38(%rdx),%rcx
  4038a2:	c5 fa 7f 46 50       	vmovdqu %xmm0,0x50(%rsi)
  4038a7:	48 89 4e 38          	mov    %rcx,0x38(%rsi)
  4038ab:	48 8d 4e 08          	lea    0x8(%rsi),%rcx
  4038af:	48 89 4e 40          	mov    %rcx,0x40(%rsi)
  4038b3:	48 8d 4e 50          	lea    0x50(%rsi),%rcx
  4038b7:	48 89 4e 48          	mov    %rcx,0x48(%rsi)
  4038bb:	74 04                	je     4038c1 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x531>
  4038bd:	f0 83 00 01          	lock addl $0x1,(%rax)
  4038c1:	83 7a 04 02          	cmpl   $0x2,0x4(%rdx)
  4038c5:	0f 8f 0f 03 00 00    	jg     403bda <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x84a>
  4038cb:	48 8b 4a 48          	mov    0x48(%rdx),%rcx
  4038cf:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  4038d4:	48 8b 09             	mov    (%rcx),%rcx
  4038d7:	48 8b 46 48          	mov    0x48(%rsi),%rax
  4038db:	48 89 08             	mov    %rcx,(%rax)
  4038de:	48 8b 52 48          	mov    0x48(%rdx),%rdx
  4038e2:	48 8b 46 48          	mov    0x48(%rsi),%rax
  4038e6:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  4038ea:	48 89 50 08          	mov    %rdx,0x8(%rax)
  4038ee:	4d 8b 27             	mov    (%r15),%r12
  4038f1:	48 8b 5c 24 18       	mov    0x18(%rsp),%rbx
  4038f6:	4d 39 e6             	cmp    %r12,%r14
  4038f9:	75 36                	jne    403931 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x5a1>
  4038fb:	e9 16 01 00 00       	jmpq   403a16 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x686>
  403900:	49 8b 54 24 48       	mov    0x48(%r12),%rdx
  403905:	48 8b 43 48          	mov    0x48(%rbx),%rax
  403909:	48 8b 12             	mov    (%rdx),%rdx
  40390c:	48 89 10             	mov    %rdx,(%rax)
  40390f:	49 8b 54 24 48       	mov    0x48(%r12),%rdx
  403914:	48 8b 43 48          	mov    0x48(%rbx),%rax
  403918:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  40391c:	48 89 50 08          	mov    %rdx,0x8(%rax)
  403920:	49 83 c4 60          	add    $0x60,%r12
  403924:	48 83 c3 60          	add    $0x60,%rbx
  403928:	4d 39 e6             	cmp    %r12,%r14
  40392b:	0f 84 e5 00 00 00    	je     403a16 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x686>
  403931:	48 85 db             	test   %rbx,%rbx
  403934:	74 ea                	je     403920 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x590>
  403936:	41 8b 04 24          	mov    (%r12),%eax
  40393a:	89 03                	mov    %eax,(%rbx)
  40393c:	41 8b 44 24 04       	mov    0x4(%r12),%eax
  403941:	89 43 04             	mov    %eax,0x4(%rbx)
  403944:	41 8b 44 24 08       	mov    0x8(%r12),%eax
  403949:	89 43 08             	mov    %eax,0x8(%rbx)
  40394c:	41 8b 44 24 0c       	mov    0xc(%r12),%eax
  403951:	89 43 0c             	mov    %eax,0xc(%rbx)
  403954:	49 8b 44 24 10       	mov    0x10(%r12),%rax
  403959:	48 89 43 10          	mov    %rax,0x10(%rbx)
  40395d:	49 8b 44 24 18       	mov    0x18(%r12),%rax
  403962:	48 89 43 18          	mov    %rax,0x18(%rbx)
  403966:	49 8b 54 24 20       	mov    0x20(%r12),%rdx
  40396b:	48 85 c0             	test   %rax,%rax
  40396e:	48 89 53 20          	mov    %rdx,0x20(%rbx)
  403972:	49 8b 54 24 28       	mov    0x28(%r12),%rdx
  403977:	48 89 53 28          	mov    %rdx,0x28(%rbx)
  40397b:	49 8b 54 24 30       	mov    0x30(%r12),%rdx
  403980:	48 89 53 30          	mov    %rdx,0x30(%rbx)
  403984:	49 8b 54 24 38       	mov    0x38(%r12),%rdx
  403989:	48 c7 43 58 00 00 00 	movq   $0x0,0x58(%rbx)
  403990:	00 
  403991:	48 c7 43 50 00 00 00 	movq   $0x0,0x50(%rbx)
  403998:	00 
  403999:	48 89 53 38          	mov    %rdx,0x38(%rbx)
  40399d:	48 8d 53 08          	lea    0x8(%rbx),%rdx
  4039a1:	48 89 53 40          	mov    %rdx,0x40(%rbx)
  4039a5:	48 8d 53 50          	lea    0x50(%rbx),%rdx
  4039a9:	48 89 53 48          	mov    %rdx,0x48(%rbx)
  4039ad:	74 04                	je     4039b3 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x623>
  4039af:	f0 83 00 01          	lock addl $0x1,(%rax)
  4039b3:	41 83 7c 24 04 02    	cmpl   $0x2,0x4(%r12)
  4039b9:	0f 8e 41 ff ff ff    	jle    403900 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x570>
  4039bf:	c7 43 04 00 00 00 00 	movl   $0x0,0x4(%rbx)
  4039c6:	4c 89 e6             	mov    %r12,%rsi
  4039c9:	48 89 df             	mov    %rbx,%rdi
  4039cc:	e8 2f db ff ff       	callq  401500 <_ZN2cv3Mat8copySizeERKS0_@plt>
  4039d1:	e9 4a ff ff ff       	jmpq   403920 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x590>
  4039d6:	48 8d 34 3f          	lea    (%rdi,%rdi,1),%rsi
  4039da:	4c 89 f3             	mov    %r14,%rbx
  4039dd:	48 29 c3             	sub    %rax,%rbx
  4039e0:	48 c1 fb 05          	sar    $0x5,%rbx
  4039e4:	48 39 f7             	cmp    %rsi,%rdi
  4039e7:	0f 86 79 02 00 00    	jbe    403c66 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x8d6>
  4039ed:	48 b8 ab aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaab,%rax
  4039f4:	aa aa aa 
  4039f7:	48 c7 c7 c0 ff ff ff 	mov    $0xffffffffffffffc0,%rdi
  4039fe:	48 0f af d8          	imul   %rax,%rbx
  403a02:	48 b8 aa aa aa aa aa 	movabs $0x2aaaaaaaaaaaaaa,%rax
  403a09:	aa aa 02 
  403a0c:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  403a11:	e9 22 fe ff ff       	jmpq   403838 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x4a8>
  403a16:	49 8b 77 08          	mov    0x8(%r15),%rsi
  403a1a:	48 8d 43 60          	lea    0x60(%rbx),%rax
  403a1e:	4d 89 f5             	mov    %r14,%r13
  403a21:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  403a26:	49 89 c4             	mov    %rax,%r12
  403a29:	49 39 f6             	cmp    %rsi,%r14
  403a2c:	48 89 74 24 20       	mov    %rsi,0x20(%rsp)
  403a31:	75 40                	jne    403a73 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x6e3>
  403a33:	e9 fb 00 00 00       	jmpq   403b33 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7a3>
  403a38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  403a3f:	00 
  403a40:	49 8b 55 48          	mov    0x48(%r13),%rdx
  403a44:	49 8b 44 24 48       	mov    0x48(%r12),%rax
  403a49:	48 8b 12             	mov    (%rdx),%rdx
  403a4c:	48 89 10             	mov    %rdx,(%rax)
  403a4f:	49 8b 55 48          	mov    0x48(%r13),%rdx
  403a53:	49 8b 44 24 48       	mov    0x48(%r12),%rax
  403a58:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  403a5c:	48 89 50 08          	mov    %rdx,0x8(%rax)
  403a60:	49 83 c4 60          	add    $0x60,%r12
  403a64:	49 83 c5 60          	add    $0x60,%r13
  403a68:	4c 39 6c 24 20       	cmp    %r13,0x20(%rsp)
  403a6d:	0f 84 bc 00 00 00    	je     403b2f <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x79f>
  403a73:	4d 85 e4             	test   %r12,%r12
  403a76:	74 e8                	je     403a60 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x6d0>
  403a78:	41 8b 45 00          	mov    0x0(%r13),%eax
  403a7c:	48 8b 74 24 28       	mov    0x28(%rsp),%rsi
  403a81:	49 8d 4c 24 08       	lea    0x8(%r12),%rcx
  403a86:	41 89 04 24          	mov    %eax,(%r12)
  403a8a:	41 8b 45 04          	mov    0x4(%r13),%eax
  403a8e:	41 89 44 24 04       	mov    %eax,0x4(%r12)
  403a93:	41 8b 45 08          	mov    0x8(%r13),%eax
  403a97:	41 89 44 24 08       	mov    %eax,0x8(%r12)
  403a9c:	41 8b 45 0c          	mov    0xc(%r13),%eax
  403aa0:	41 89 44 24 0c       	mov    %eax,0xc(%r12)
  403aa5:	49 8b 45 10          	mov    0x10(%r13),%rax
  403aa9:	49 89 44 24 10       	mov    %rax,0x10(%r12)
  403aae:	49 8b 45 18          	mov    0x18(%r13),%rax
  403ab2:	49 89 44 24 18       	mov    %rax,0x18(%r12)
  403ab7:	49 8b 55 20          	mov    0x20(%r13),%rdx
  403abb:	49 89 54 24 20       	mov    %rdx,0x20(%r12)
  403ac0:	49 8b 55 28          	mov    0x28(%r13),%rdx
  403ac4:	49 89 54 24 28       	mov    %rdx,0x28(%r12)
  403ac9:	49 8b 55 30          	mov    0x30(%r13),%rdx
  403acd:	49 89 54 24 30       	mov    %rdx,0x30(%r12)
  403ad2:	49 8b 55 38          	mov    0x38(%r13),%rdx
  403ad6:	49 89 54 24 38       	mov    %rdx,0x38(%r12)
  403adb:	4c 89 e2             	mov    %r12,%rdx
  403ade:	48 29 da             	sub    %rbx,%rdx
  403ae1:	48 85 c0             	test   %rax,%rax
  403ae4:	48 89 4c 16 e0       	mov    %rcx,-0x20(%rsi,%rdx,1)
  403ae9:	49 8d 4c 24 50       	lea    0x50(%r12),%rcx
  403aee:	48 89 4c 16 e8       	mov    %rcx,-0x18(%rsi,%rdx,1)
  403af3:	49 c7 44 24 58 00 00 	movq   $0x0,0x58(%r12)
  403afa:	00 00 
  403afc:	49 c7 44 24 50 00 00 	movq   $0x0,0x50(%r12)
  403b03:	00 00 
  403b05:	74 04                	je     403b0b <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x77b>
  403b07:	f0 83 00 01          	lock addl $0x1,(%rax)
  403b0b:	41 83 7d 04 02       	cmpl   $0x2,0x4(%r13)
  403b10:	0f 8e 2a ff ff ff    	jle    403a40 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x6b0>
  403b16:	41 c7 44 24 04 00 00 	movl   $0x0,0x4(%r12)
  403b1d:	00 00 
  403b1f:	4c 89 ee             	mov    %r13,%rsi
  403b22:	4c 89 e7             	mov    %r12,%rdi
  403b25:	e8 d6 d9 ff ff       	callq  401500 <_ZN2cv3Mat8copySizeERKS0_@plt>
  403b2a:	e9 31 ff ff ff       	jmpq   403a60 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x6d0>
  403b2f:	4d 8b 77 08          	mov    0x8(%r15),%r14
  403b33:	49 8b 1f             	mov    (%r15),%rbx
  403b36:	4c 89 f7             	mov    %r14,%rdi
  403b39:	4c 39 f3             	cmp    %r14,%rbx
  403b3c:	75 4f                	jne    403b8d <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7fd>
  403b3e:	eb 69                	jmp    403ba9 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x819>
  403b40:	48 8b 43 40          	mov    0x40(%rbx),%rax
  403b44:	48 c7 43 30 00 00 00 	movq   $0x0,0x30(%rbx)
  403b4b:	00 
  403b4c:	48 c7 43 28 00 00 00 	movq   $0x0,0x28(%rbx)
  403b53:	00 
  403b54:	48 c7 43 20 00 00 00 	movq   $0x0,0x20(%rbx)
  403b5b:	00 
  403b5c:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
  403b63:	00 
  403b64:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  403b6a:	48 8b 7b 48          	mov    0x48(%rbx),%rdi
  403b6e:	48 8d 43 50          	lea    0x50(%rbx),%rax
  403b72:	48 c7 43 18 00 00 00 	movq   $0x0,0x18(%rbx)
  403b79:	00 
  403b7a:	48 39 c7             	cmp    %rax,%rdi
  403b7d:	74 05                	je     403b84 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7f4>
  403b7f:	e8 4c db ff ff       	callq  4016d0 <_ZN2cv8fastFreeEPv@plt>
  403b84:	48 83 c3 60          	add    $0x60,%rbx
  403b88:	4c 39 f3             	cmp    %r14,%rbx
  403b8b:	74 19                	je     403ba6 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x816>
  403b8d:	48 8b 43 18          	mov    0x18(%rbx),%rax
  403b91:	48 85 c0             	test   %rax,%rax
  403b94:	74 aa                	je     403b40 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7b0>
  403b96:	f0 83 28 01          	lock subl $0x1,(%rax)
  403b9a:	75 a4                	jne    403b40 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7b0>
  403b9c:	48 89 df             	mov    %rbx,%rdi
  403b9f:	e8 1c da ff ff       	callq  4015c0 <_ZN2cv3Mat10deallocateEv@plt>
  403ba4:	eb 9a                	jmp    403b40 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7b0>
  403ba6:	49 8b 3f             	mov    (%r15),%rdi
  403ba9:	48 85 ff             	test   %rdi,%rdi
  403bac:	74 07                	je     403bb5 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x825>
  403bae:	66 90                	xchg   %ax,%ax
  403bb0:	e8 1b da ff ff       	callq  4015d0 <_ZdlPv@plt>
  403bb5:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  403bba:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
  403bbf:	4d 89 67 08          	mov    %r12,0x8(%r15)
  403bc3:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  403bc7:	49 89 37             	mov    %rsi,(%r15)
  403bca:	48 c1 e0 05          	shl    $0x5,%rax
  403bce:	48 01 f0             	add    %rsi,%rax
  403bd1:	49 89 47 10          	mov    %rax,0x10(%r15)
  403bd5:	e9 5b fb ff ff       	jmpq   403735 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x3a5>
  403bda:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  403bdf:	48 89 d6             	mov    %rdx,%rsi
  403be2:	c7 40 04 00 00 00 00 	movl   $0x0,0x4(%rax)
  403be9:	48 89 c7             	mov    %rax,%rdi
  403bec:	e8 0f d9 ff ff       	callq  401500 <_ZN2cv3Mat8copySizeERKS0_@plt>
  403bf1:	e9 f8 fc ff ff       	jmpq   4038ee <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x55e>
  403bf6:	48 89 c7             	mov    %rax,%rdi
  403bf9:	c5 f8 77             	vzeroupper 
  403bfc:	e8 df d8 ff ff       	callq  4014e0 <__cxa_begin_catch@plt>
  403c01:	4c 39 64 24 28       	cmp    %r12,0x28(%rsp)
  403c06:	74 16                	je     403c1e <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x88e>
  403c08:	48 8b 5c 24 28       	mov    0x28(%rsp),%rbx
  403c0d:	48 89 df             	mov    %rbx,%rdi
  403c10:	e8 1b f6 ff ff       	callq  403230 <_ZN2cv3MatD1Ev>
  403c15:	48 83 c3 60          	add    $0x60,%rbx
  403c19:	4c 39 e3             	cmp    %r12,%rbx
  403c1c:	75 ef                	jne    403c0d <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x87d>
  403c1e:	e8 6d da ff ff       	callq  401690 <__cxa_rethrow@plt>
  403c23:	48 89 c3             	mov    %rax,%rbx
  403c26:	4c 89 ef             	mov    %r13,%rdi
  403c29:	c5 f8 77             	vzeroupper 
  403c2c:	e8 ff f5 ff ff       	callq  403230 <_ZN2cv3MatD1Ev>
  403c31:	48 89 df             	mov    %rbx,%rdi
  403c34:	e8 e7 d9 ff ff       	callq  401620 <_Unwind_Resume@plt>
  403c39:	48 89 c7             	mov    %rax,%rdi
  403c3c:	c5 f8 77             	vzeroupper 
  403c3f:	e8 9c d8 ff ff       	callq  4014e0 <__cxa_begin_catch@plt>
  403c44:	48 39 5c 24 18       	cmp    %rbx,0x18(%rsp)
  403c49:	74 16                	je     403c61 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x8d1>
  403c4b:	4c 8b 64 24 18       	mov    0x18(%rsp),%r12
  403c50:	4c 89 e7             	mov    %r12,%rdi
  403c53:	e8 d8 f5 ff ff       	callq  403230 <_ZN2cv3MatD1Ev>
  403c58:	49 83 c4 60          	add    $0x60,%r12
  403c5c:	4c 39 e3             	cmp    %r12,%rbx
  403c5f:	75 ef                	jne    403c50 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x8c0>
  403c61:	e8 2a da ff ff       	callq  401690 <__cxa_rethrow@plt>
  403c66:	48 bf aa aa aa aa aa 	movabs $0x2aaaaaaaaaaaaaa,%rdi
  403c6d:	aa aa 02 
  403c70:	48 39 fe             	cmp    %rdi,%rsi
  403c73:	0f 87 74 fd ff ff    	ja     4039ed <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x65d>
  403c79:	48 0f af d9          	imul   %rcx,%rbx
  403c7d:	48 85 f6             	test   %rsi,%rsi
  403c80:	0f 85 a5 fb ff ff    	jne    40382b <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x49b>
  403c86:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
  403c8d:	00 00 
  403c8f:	48 c7 44 24 18 00 00 	movq   $0x0,0x18(%rsp)
  403c96:	00 00 
  403c98:	e9 af fb ff ff       	jmpq   40384c <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x4bc>
  403c9d:	48 89 c7             	mov    %rax,%rdi
  403ca0:	c5 f8 77             	vzeroupper 
  403ca3:	e8 38 d8 ff ff       	callq  4014e0 <__cxa_begin_catch@plt>
  403ca8:	48 83 7c 24 18 00    	cmpq   $0x0,0x18(%rsp)
  403cae:	74 69                	je     403d19 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x989>
  403cb0:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
  403cb5:	e8 16 d9 ff ff       	callq  4015d0 <_ZdlPv@plt>
  403cba:	e8 d1 d9 ff ff       	callq  401690 <__cxa_rethrow@plt>
  403cbf:	48 89 c3             	mov    %rax,%rbx
  403cc2:	c5 f8 77             	vzeroupper 
  403cc5:	e8 a6 d9 ff ff       	callq  401670 <__cxa_end_catch@plt>
  403cca:	48 89 df             	mov    %rbx,%rdi
  403ccd:	e8 0e d8 ff ff       	callq  4014e0 <__cxa_begin_catch@plt>
  403cd2:	48 83 7c 24 28 00    	cmpq   $0x0,0x28(%rsp)
  403cd8:	74 3f                	je     403d19 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x989>
  403cda:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  403cdf:	48 39 44 24 28       	cmp    %rax,0x28(%rsp)
  403ce4:	74 ca                	je     403cb0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x920>
  403ce6:	48 89 c3             	mov    %rax,%rbx
  403ce9:	48 89 df             	mov    %rbx,%rdi
  403cec:	e8 3f f5 ff ff       	callq  403230 <_ZN2cv3MatD1Ev>
  403cf1:	48 83 c3 60          	add    $0x60,%rbx
  403cf5:	48 39 5c 24 28       	cmp    %rbx,0x28(%rsp)
  403cfa:	75 ed                	jne    403ce9 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x959>
  403cfc:	48 83 7c 24 18 00    	cmpq   $0x0,0x18(%rsp)
  403d02:	75 ac                	jne    403cb0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x920>
  403d04:	eb b4                	jmp    403cba <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x92a>
  403d06:	48 89 c3             	mov    %rax,%rbx
  403d09:	c5 f8 77             	vzeroupper 
  403d0c:	e8 5f d9 ff ff       	callq  401670 <__cxa_end_catch@plt>
  403d11:	48 89 df             	mov    %rbx,%rdi
  403d14:	e8 c7 d7 ff ff       	callq  4014e0 <__cxa_begin_catch@plt>
  403d19:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  403d1e:	e8 0d f5 ff ff       	callq  403230 <_ZN2cv3MatD1Ev>
  403d23:	eb d7                	jmp    403cfc <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x96c>
  403d25:	48 89 c3             	mov    %rax,%rbx
  403d28:	c5 f8 77             	vzeroupper 
  403d2b:	e8 40 d9 ff ff       	callq  401670 <__cxa_end_catch@plt>
  403d30:	48 89 df             	mov    %rbx,%rdi
  403d33:	e8 e8 d8 ff ff       	callq  401620 <_Unwind_Resume@plt>
  403d38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  403d3f:	00 

0000000000403d40 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag>:
  403d40:	55                   	push   %rbp
  403d41:	48 89 e5             	mov    %rsp,%rbp
  403d44:	41 56                	push   %r14
  403d46:	41 55                	push   %r13
  403d48:	41 54                	push   %r12
  403d4a:	49 89 d4             	mov    %rdx,%r12
  403d4d:	49 29 f4             	sub    %rsi,%r12
  403d50:	53                   	push   %rbx
  403d51:	48 89 fb             	mov    %rdi,%rbx
  403d54:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  403d58:	48 83 ec 10          	sub    $0x10,%rsp
  403d5c:	48 8b 07             	mov    (%rdi),%rax
  403d5f:	48 8b 4f 10          	mov    0x10(%rdi),%rcx
  403d63:	48 29 c1             	sub    %rax,%rcx
  403d66:	48 c1 f9 02          	sar    $0x2,%rcx
  403d6a:	49 39 cc             	cmp    %rcx,%r12
  403d6d:	0f 87 0d 04 00 00    	ja     404180 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x440>
  403d73:	4c 8b 4f 08          	mov    0x8(%rdi),%r9
  403d77:	4c 89 cf             	mov    %r9,%rdi
  403d7a:	48 29 c7             	sub    %rax,%rdi
  403d7d:	48 c1 ff 02          	sar    $0x2,%rdi
  403d81:	49 39 fc             	cmp    %rdi,%r12
  403d84:	0f 87 2e 01 00 00    	ja     403eb8 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x178>
  403d8a:	4d 85 e4             	test   %r12,%r12
  403d8d:	0f 8e a5 05 00 00    	jle    404338 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5f8>
  403d93:	4a 8d 3c a0          	lea    (%rax,%r12,4),%rdi
  403d97:	4a 8d 0c 26          	lea    (%rsi,%r12,1),%rcx
  403d9b:	48 39 fe             	cmp    %rdi,%rsi
  403d9e:	41 0f 93 c0          	setae  %r8b
  403da2:	48 39 c8             	cmp    %rcx,%rax
  403da5:	0f 93 c2             	setae  %dl
  403da8:	41 08 d0             	or     %dl,%r8b
  403dab:	0f 84 8f 03 00 00    	je     404140 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x400>
  403db1:	49 83 fc 1f          	cmp    $0x1f,%r12
  403db5:	0f 86 85 03 00 00    	jbe    404140 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x400>
  403dbb:	4d 89 e0             	mov    %r12,%r8
  403dbe:	48 8d 50 20          	lea    0x20(%rax),%rdx
  403dc2:	45 31 c9             	xor    %r9d,%r9d
  403dc5:	49 c1 e8 05          	shr    $0x5,%r8
  403dc9:	4c 89 c1             	mov    %r8,%rcx
  403dcc:	48 c1 e1 05          	shl    $0x5,%rcx
  403dd0:	4d 89 ca             	mov    %r9,%r10
  403dd3:	49 83 c1 01          	add    $0x1,%r9
  403dd7:	48 83 ea 80          	sub    $0xffffffffffffff80,%rdx
  403ddb:	49 c1 e2 05          	shl    $0x5,%r10
  403ddf:	c4 a1 7a 6f 04 16    	vmovdqu (%rsi,%r10,1),%xmm0
  403de5:	c4 a3 7d 38 44 16 10 	vinserti128 $0x1,0x10(%rsi,%r10,1),%ymm0,%ymm0
  403dec:	01 
  403ded:	c4 e2 7d 30 c8       	vpmovzxbw %xmm0,%ymm1
  403df2:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  403df8:	c4 e2 7d 23 d1       	vpmovsxwd %xmm1,%ymm2
  403dfd:	c4 e3 7d 39 c9 01    	vextracti128 $0x1,%ymm1,%xmm1
  403e03:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  403e08:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  403e0c:	c5 f8 11 92 60 ff ff 	vmovups %xmm2,-0xa0(%rdx)
  403e13:	ff 
  403e14:	c4 e2 7d 23 c9       	vpmovsxwd %xmm1,%ymm1
  403e19:	c4 e3 7d 19 92 70 ff 	vextractf128 $0x1,%ymm2,-0x90(%rdx)
  403e20:	ff ff 01 
  403e23:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  403e27:	c5 f8 11 4a 80       	vmovups %xmm1,-0x80(%rdx)
  403e2c:	c4 e3 7d 19 4a 90 01 	vextractf128 $0x1,%ymm1,-0x70(%rdx)
  403e33:	c4 e2 7d 23 c8       	vpmovsxwd %xmm0,%ymm1
  403e38:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  403e3e:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  403e42:	c5 f8 11 4a a0       	vmovups %xmm1,-0x60(%rdx)
  403e47:	c4 e3 7d 19 4a b0 01 	vextractf128 $0x1,%ymm1,-0x50(%rdx)
  403e4e:	c4 e2 7d 23 c0       	vpmovsxwd %xmm0,%ymm0
  403e53:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  403e57:	c5 f8 11 42 c0       	vmovups %xmm0,-0x40(%rdx)
  403e5c:	c4 e3 7d 19 42 d0 01 	vextractf128 $0x1,%ymm0,-0x30(%rdx)
  403e63:	4d 39 c1             	cmp    %r8,%r9
  403e66:	0f 82 64 ff ff ff    	jb     403dd0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x90>
  403e6c:	48 8d 14 88          	lea    (%rax,%rcx,4),%rdx
  403e70:	4c 89 e0             	mov    %r12,%rax
  403e73:	48 01 ce             	add    %rcx,%rsi
  403e76:	48 29 c8             	sub    %rcx,%rax
  403e79:	49 39 cc             	cmp    %rcx,%r12
  403e7c:	74 1f                	je     403e9d <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x15d>
  403e7e:	66 90                	xchg   %ax,%ax
  403e80:	0f b6 0e             	movzbl (%rsi),%ecx
  403e83:	48 83 e8 01          	sub    $0x1,%rax
  403e87:	48 83 c6 01          	add    $0x1,%rsi
  403e8b:	48 83 c2 04          	add    $0x4,%rdx
  403e8f:	c5 fa 2a c1          	vcvtsi2ss %ecx,%xmm0,%xmm0
  403e93:	c5 fa 11 42 fc       	vmovss %xmm0,-0x4(%rdx)
  403e98:	48 85 c0             	test   %rax,%rax
  403e9b:	7f e3                	jg     403e80 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x140>
  403e9d:	c5 f8 77             	vzeroupper 
  403ea0:	48 89 7b 08          	mov    %rdi,0x8(%rbx)
  403ea4:	48 8d 65 e0          	lea    -0x20(%rbp),%rsp
  403ea8:	5b                   	pop    %rbx
  403ea9:	41 5c                	pop    %r12
  403eab:	41 5d                	pop    %r13
  403ead:	41 5e                	pop    %r14
  403eaf:	5d                   	pop    %rbp
  403eb0:	c3                   	retq   
  403eb1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  403eb8:	48 01 f7             	add    %rsi,%rdi
  403ebb:	49 89 fa             	mov    %rdi,%r10
  403ebe:	49 29 f2             	sub    %rsi,%r10
  403ec1:	4d 85 d2             	test   %r10,%r10
  403ec4:	0f 8e 14 01 00 00    	jle    403fde <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x29e>
  403eca:	4a 8d 0c 90          	lea    (%rax,%r10,4),%rcx
  403ece:	4e 8d 04 16          	lea    (%rsi,%r10,1),%r8
  403ed2:	48 39 ce             	cmp    %rcx,%rsi
  403ed5:	41 0f 93 c3          	setae  %r11b
  403ed9:	4c 39 c0             	cmp    %r8,%rax
  403edc:	0f 93 c1             	setae  %cl
  403edf:	41 08 cb             	or     %cl,%r11b
  403ee2:	0f 84 30 02 00 00    	je     404118 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3d8>
  403ee8:	49 83 fa 1f          	cmp    $0x1f,%r10
  403eec:	0f 86 26 02 00 00    	jbe    404118 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3d8>
  403ef2:	4d 89 d0             	mov    %r10,%r8
  403ef5:	48 8d 48 20          	lea    0x20(%rax),%rcx
  403ef9:	45 31 e4             	xor    %r12d,%r12d
  403efc:	49 c1 e8 05          	shr    $0x5,%r8
  403f00:	4d 89 c3             	mov    %r8,%r11
  403f03:	49 c1 e3 05          	shl    $0x5,%r11
  403f07:	4d 89 e5             	mov    %r12,%r13
  403f0a:	49 83 c4 01          	add    $0x1,%r12
  403f0e:	48 83 e9 80          	sub    $0xffffffffffffff80,%rcx
  403f12:	49 c1 e5 05          	shl    $0x5,%r13
  403f16:	c4 a1 7a 6f 04 2e    	vmovdqu (%rsi,%r13,1),%xmm0
  403f1c:	c4 a3 7d 38 44 2e 10 	vinserti128 $0x1,0x10(%rsi,%r13,1),%ymm0,%ymm0
  403f23:	01 
  403f24:	c4 e2 7d 30 c8       	vpmovzxbw %xmm0,%ymm1
  403f29:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  403f2f:	c4 e2 7d 23 d1       	vpmovsxwd %xmm1,%ymm2
  403f34:	c4 e3 7d 39 c9 01    	vextracti128 $0x1,%ymm1,%xmm1
  403f3a:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  403f3f:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  403f43:	c5 f8 11 91 60 ff ff 	vmovups %xmm2,-0xa0(%rcx)
  403f4a:	ff 
  403f4b:	c4 e2 7d 23 c9       	vpmovsxwd %xmm1,%ymm1
  403f50:	c4 e3 7d 19 91 70 ff 	vextractf128 $0x1,%ymm2,-0x90(%rcx)
  403f57:	ff ff 01 
  403f5a:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  403f5e:	c5 f8 11 49 80       	vmovups %xmm1,-0x80(%rcx)
  403f63:	c4 e3 7d 19 49 90 01 	vextractf128 $0x1,%ymm1,-0x70(%rcx)
  403f6a:	c4 e2 7d 23 c8       	vpmovsxwd %xmm0,%ymm1
  403f6f:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  403f75:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  403f79:	c5 f8 11 49 a0       	vmovups %xmm1,-0x60(%rcx)
  403f7e:	c4 e3 7d 19 49 b0 01 	vextractf128 $0x1,%ymm1,-0x50(%rcx)
  403f85:	c4 e2 7d 23 c0       	vpmovsxwd %xmm0,%ymm0
  403f8a:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  403f8e:	c5 f8 11 41 c0       	vmovups %xmm0,-0x40(%rcx)
  403f93:	c4 e3 7d 19 41 d0 01 	vextractf128 $0x1,%ymm0,-0x30(%rcx)
  403f9a:	4d 39 c4             	cmp    %r8,%r12
  403f9d:	0f 82 64 ff ff ff    	jb     403f07 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x1c7>
  403fa3:	4a 8d 0c 98          	lea    (%rax,%r11,4),%rcx
  403fa7:	4c 89 d0             	mov    %r10,%rax
  403faa:	4e 8d 04 1e          	lea    (%rsi,%r11,1),%r8
  403fae:	4c 29 d8             	sub    %r11,%rax
  403fb1:	4d 39 da             	cmp    %r11,%r10
  403fb4:	74 28                	je     403fde <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x29e>
  403fb6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  403fbd:	00 00 00 
  403fc0:	41 0f b6 30          	movzbl (%r8),%esi
  403fc4:	48 83 e8 01          	sub    $0x1,%rax
  403fc8:	49 83 c0 01          	add    $0x1,%r8
  403fcc:	48 83 c1 04          	add    $0x4,%rcx
  403fd0:	c5 fa 2a c6          	vcvtsi2ss %esi,%xmm0,%xmm0
  403fd4:	c5 fa 11 41 fc       	vmovss %xmm0,-0x4(%rcx)
  403fd9:	48 85 c0             	test   %rax,%rax
  403fdc:	7f e2                	jg     403fc0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x280>
  403fde:	48 29 fa             	sub    %rdi,%rdx
  403fe1:	4d 89 c8             	mov    %r9,%r8
  403fe4:	48 85 d2             	test   %rdx,%rdx
  403fe7:	0f 8e 10 01 00 00    	jle    4040fd <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3bd>
  403fed:	4d 8d 04 91          	lea    (%r9,%rdx,4),%r8
  403ff1:	48 8d 0c 17          	lea    (%rdi,%rdx,1),%rcx
  403ff5:	4c 39 c7             	cmp    %r8,%rdi
  403ff8:	40 0f 93 c6          	setae  %sil
  403ffc:	49 39 c9             	cmp    %rcx,%r9
  403fff:	0f 93 c0             	setae  %al
  404002:	40 08 c6             	or     %al,%sil
  404005:	0f 84 55 01 00 00    	je     404160 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x420>
  40400b:	48 83 fa 1f          	cmp    $0x1f,%rdx
  40400f:	0f 86 4b 01 00 00    	jbe    404160 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x420>
  404015:	48 89 d1             	mov    %rdx,%rcx
  404018:	49 8d 41 20          	lea    0x20(%r9),%rax
  40401c:	31 f6                	xor    %esi,%esi
  40401e:	48 c1 e9 05          	shr    $0x5,%rcx
  404022:	49 89 ca             	mov    %rcx,%r10
  404025:	49 c1 e2 05          	shl    $0x5,%r10
  404029:	49 89 f3             	mov    %rsi,%r11
  40402c:	48 83 c6 01          	add    $0x1,%rsi
  404030:	48 83 e8 80          	sub    $0xffffffffffffff80,%rax
  404034:	49 c1 e3 05          	shl    $0x5,%r11
  404038:	c4 a1 7a 6f 04 1f    	vmovdqu (%rdi,%r11,1),%xmm0
  40403e:	c4 a3 7d 38 44 1f 10 	vinserti128 $0x1,0x10(%rdi,%r11,1),%ymm0,%ymm0
  404045:	01 
  404046:	c4 e2 7d 30 c8       	vpmovzxbw %xmm0,%ymm1
  40404b:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  404051:	c4 e2 7d 23 d1       	vpmovsxwd %xmm1,%ymm2
  404056:	c4 e3 7d 39 c9 01    	vextracti128 $0x1,%ymm1,%xmm1
  40405c:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  404061:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  404065:	c5 f8 11 90 60 ff ff 	vmovups %xmm2,-0xa0(%rax)
  40406c:	ff 
  40406d:	c4 e2 7d 23 c9       	vpmovsxwd %xmm1,%ymm1
  404072:	c4 e3 7d 19 90 70 ff 	vextractf128 $0x1,%ymm2,-0x90(%rax)
  404079:	ff ff 01 
  40407c:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  404080:	c5 f8 11 48 80       	vmovups %xmm1,-0x80(%rax)
  404085:	c4 e3 7d 19 48 90 01 	vextractf128 $0x1,%ymm1,-0x70(%rax)
  40408c:	c4 e2 7d 23 c8       	vpmovsxwd %xmm0,%ymm1
  404091:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  404097:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  40409b:	c5 f8 11 48 a0       	vmovups %xmm1,-0x60(%rax)
  4040a0:	c4 e3 7d 19 48 b0 01 	vextractf128 $0x1,%ymm1,-0x50(%rax)
  4040a7:	c4 e2 7d 23 c0       	vpmovsxwd %xmm0,%ymm0
  4040ac:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  4040b0:	c5 f8 11 40 c0       	vmovups %xmm0,-0x40(%rax)
  4040b5:	c4 e3 7d 19 40 d0 01 	vextractf128 $0x1,%ymm0,-0x30(%rax)
  4040bc:	48 39 ce             	cmp    %rcx,%rsi
  4040bf:	0f 82 64 ff ff ff    	jb     404029 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x2e9>
  4040c5:	48 89 d0             	mov    %rdx,%rax
  4040c8:	4a 8d 34 17          	lea    (%rdi,%r10,1),%rsi
  4040cc:	4b 8d 0c 91          	lea    (%r9,%r10,4),%rcx
  4040d0:	4c 29 d0             	sub    %r10,%rax
  4040d3:	4c 39 d2             	cmp    %r10,%rdx
  4040d6:	74 25                	je     4040fd <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3bd>
  4040d8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4040df:	00 
  4040e0:	0f b6 16             	movzbl (%rsi),%edx
  4040e3:	48 83 e8 01          	sub    $0x1,%rax
  4040e7:	48 83 c6 01          	add    $0x1,%rsi
  4040eb:	48 83 c1 04          	add    $0x4,%rcx
  4040ef:	c5 fa 2a c2          	vcvtsi2ss %edx,%xmm0,%xmm0
  4040f3:	c5 fa 11 41 fc       	vmovss %xmm0,-0x4(%rcx)
  4040f8:	48 85 c0             	test   %rax,%rax
  4040fb:	7f e3                	jg     4040e0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3a0>
  4040fd:	4c 89 43 08          	mov    %r8,0x8(%rbx)
  404101:	c5 f8 77             	vzeroupper 
  404104:	48 8d 65 e0          	lea    -0x20(%rbp),%rsp
  404108:	5b                   	pop    %rbx
  404109:	41 5c                	pop    %r12
  40410b:	41 5d                	pop    %r13
  40410d:	41 5e                	pop    %r14
  40410f:	5d                   	pop    %rbp
  404110:	c3                   	retq   
  404111:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  404118:	0f b6 0e             	movzbl (%rsi),%ecx
  40411b:	48 83 c6 01          	add    $0x1,%rsi
  40411f:	48 83 c0 04          	add    $0x4,%rax
  404123:	c5 fa 2a c1          	vcvtsi2ss %ecx,%xmm0,%xmm0
  404127:	c5 fa 11 40 fc       	vmovss %xmm0,-0x4(%rax)
  40412c:	49 39 f0             	cmp    %rsi,%r8
  40412f:	75 e7                	jne    404118 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3d8>
  404131:	e9 a8 fe ff ff       	jmpq   403fde <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x29e>
  404136:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40413d:	00 00 00 
  404140:	0f b6 16             	movzbl (%rsi),%edx
  404143:	48 83 c6 01          	add    $0x1,%rsi
  404147:	48 83 c0 04          	add    $0x4,%rax
  40414b:	c5 fa 2a c2          	vcvtsi2ss %edx,%xmm0,%xmm0
  40414f:	c5 fa 11 40 fc       	vmovss %xmm0,-0x4(%rax)
  404154:	48 39 f1             	cmp    %rsi,%rcx
  404157:	75 e7                	jne    404140 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x400>
  404159:	e9 42 fd ff ff       	jmpq   403ea0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x160>
  40415e:	66 90                	xchg   %ax,%ax
  404160:	0f b6 07             	movzbl (%rdi),%eax
  404163:	48 83 c7 01          	add    $0x1,%rdi
  404167:	49 83 c1 04          	add    $0x4,%r9
  40416b:	c5 fa 2a c0          	vcvtsi2ss %eax,%xmm0,%xmm0
  40416f:	c4 c1 7a 11 41 fc    	vmovss %xmm0,-0x4(%r9)
  404175:	48 39 f9             	cmp    %rdi,%rcx
  404178:	75 e6                	jne    404160 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x420>
  40417a:	eb 81                	jmp    4040fd <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3bd>
  40417c:	0f 1f 40 00          	nopl   0x0(%rax)
  404180:	4d 85 e4             	test   %r12,%r12
  404183:	0f 84 9f 01 00 00    	je     404328 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5e8>
  404189:	48 b8 ff ff ff ff ff 	movabs $0x3fffffffffffffff,%rax
  404190:	ff ff 3f 
  404193:	49 39 c4             	cmp    %rax,%r12
  404196:	0f 87 a4 01 00 00    	ja     404340 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x600>
  40419c:	4e 8d 34 a5 00 00 00 	lea    0x0(,%r12,4),%r14
  4041a3:	00 
  4041a4:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  4041a9:	4c 89 f7             	mov    %r14,%rdi
  4041ac:	e8 6f d3 ff ff       	callq  401520 <_Znwm@plt>
  4041b1:	4d 85 e4             	test   %r12,%r12
  4041b4:	49 89 c5             	mov    %rax,%r13
  4041b7:	48 8b 3b             	mov    (%rbx),%rdi
  4041ba:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  4041bf:	0f 8e 0d 01 00 00    	jle    4042d2 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x592>
  4041c5:	4a 8d 04 30          	lea    (%rax,%r14,1),%rax
  4041c9:	4e 8d 04 26          	lea    (%rsi,%r12,1),%r8
  4041cd:	48 39 c6             	cmp    %rax,%rsi
  4041d0:	0f 93 c2             	setae  %dl
  4041d3:	4d 39 e8             	cmp    %r13,%r8
  4041d6:	0f 96 c0             	setbe  %al
  4041d9:	08 c2                	or     %al,%dl
  4041db:	0f 84 1f 01 00 00    	je     404300 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5c0>
  4041e1:	49 83 fc 1f          	cmp    $0x1f,%r12
  4041e5:	0f 86 15 01 00 00    	jbe    404300 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5c0>
  4041eb:	4c 89 e1             	mov    %r12,%rcx
  4041ee:	49 8d 55 20          	lea    0x20(%r13),%rdx
  4041f2:	45 31 c0             	xor    %r8d,%r8d
  4041f5:	48 c1 e9 05          	shr    $0x5,%rcx
  4041f9:	48 89 c8             	mov    %rcx,%rax
  4041fc:	48 c1 e0 05          	shl    $0x5,%rax
  404200:	4d 89 c1             	mov    %r8,%r9
  404203:	49 83 c0 01          	add    $0x1,%r8
  404207:	48 83 ea 80          	sub    $0xffffffffffffff80,%rdx
  40420b:	49 c1 e1 05          	shl    $0x5,%r9
  40420f:	c4 a1 7a 6f 04 0e    	vmovdqu (%rsi,%r9,1),%xmm0
  404215:	c4 a3 7d 38 44 0e 10 	vinserti128 $0x1,0x10(%rsi,%r9,1),%ymm0,%ymm0
  40421c:	01 
  40421d:	c4 e2 7d 30 c8       	vpmovzxbw %xmm0,%ymm1
  404222:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  404228:	c4 e2 7d 23 d1       	vpmovsxwd %xmm1,%ymm2
  40422d:	c4 e3 7d 39 c9 01    	vextracti128 $0x1,%ymm1,%xmm1
  404233:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  404238:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  40423c:	c5 f8 11 92 60 ff ff 	vmovups %xmm2,-0xa0(%rdx)
  404243:	ff 
  404244:	c4 e2 7d 23 c9       	vpmovsxwd %xmm1,%ymm1
  404249:	c4 e3 7d 19 92 70 ff 	vextractf128 $0x1,%ymm2,-0x90(%rdx)
  404250:	ff ff 01 
  404253:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  404257:	c5 f8 11 4a 80       	vmovups %xmm1,-0x80(%rdx)
  40425c:	c4 e3 7d 19 4a 90 01 	vextractf128 $0x1,%ymm1,-0x70(%rdx)
  404263:	c4 e2 7d 23 c8       	vpmovsxwd %xmm0,%ymm1
  404268:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  40426e:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  404272:	c5 f8 11 4a a0       	vmovups %xmm1,-0x60(%rdx)
  404277:	c4 e3 7d 19 4a b0 01 	vextractf128 $0x1,%ymm1,-0x50(%rdx)
  40427e:	c4 e2 7d 23 c0       	vpmovsxwd %xmm0,%ymm0
  404283:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  404287:	c5 f8 11 42 c0       	vmovups %xmm0,-0x40(%rdx)
  40428c:	c4 e3 7d 19 42 d0 01 	vextractf128 $0x1,%ymm0,-0x30(%rdx)
  404293:	4c 39 c1             	cmp    %r8,%rcx
  404296:	0f 87 64 ff ff ff    	ja     404200 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x4c0>
  40429c:	4c 89 e2             	mov    %r12,%rdx
  40429f:	48 01 c6             	add    %rax,%rsi
  4042a2:	49 8d 4c 85 00       	lea    0x0(%r13,%rax,4),%rcx
  4042a7:	48 29 c2             	sub    %rax,%rdx
  4042aa:	49 39 c4             	cmp    %rax,%r12
  4042ad:	74 20                	je     4042cf <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x58f>
  4042af:	90                   	nop
  4042b0:	44 0f b6 06          	movzbl (%rsi),%r8d
  4042b4:	48 83 ea 01          	sub    $0x1,%rdx
  4042b8:	48 83 c6 01          	add    $0x1,%rsi
  4042bc:	48 83 c1 04          	add    $0x4,%rcx
  4042c0:	c4 c1 7a 2a c0       	vcvtsi2ss %r8d,%xmm0,%xmm0
  4042c5:	c5 fa 11 41 fc       	vmovss %xmm0,-0x4(%rcx)
  4042ca:	48 85 d2             	test   %rdx,%rdx
  4042cd:	7f e1                	jg     4042b0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x570>
  4042cf:	c5 f8 77             	vzeroupper 
  4042d2:	48 85 ff             	test   %rdi,%rdi
  4042d5:	74 05                	je     4042dc <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x59c>
  4042d7:	e8 f4 d2 ff ff       	callq  4015d0 <_ZdlPv@plt>
  4042dc:	4c 89 2b             	mov    %r13,(%rbx)
  4042df:	4d 01 f5             	add    %r14,%r13
  4042e2:	4c 89 6b 08          	mov    %r13,0x8(%rbx)
  4042e6:	4c 89 6b 10          	mov    %r13,0x10(%rbx)
  4042ea:	48 8d 65 e0          	lea    -0x20(%rbp),%rsp
  4042ee:	5b                   	pop    %rbx
  4042ef:	41 5c                	pop    %r12
  4042f1:	41 5d                	pop    %r13
  4042f3:	41 5e                	pop    %r14
  4042f5:	5d                   	pop    %rbp
  4042f6:	c3                   	retq   
  4042f7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4042fe:	00 00 
  404300:	4c 89 ea             	mov    %r13,%rdx
  404303:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  404308:	0f b6 0e             	movzbl (%rsi),%ecx
  40430b:	48 83 c6 01          	add    $0x1,%rsi
  40430f:	48 83 c2 04          	add    $0x4,%rdx
  404313:	c5 fa 2a c1          	vcvtsi2ss %ecx,%xmm0,%xmm0
  404317:	c5 fa 11 42 fc       	vmovss %xmm0,-0x4(%rdx)
  40431c:	49 39 f0             	cmp    %rsi,%r8
  40431f:	75 e7                	jne    404308 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5c8>
  404321:	eb af                	jmp    4042d2 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x592>
  404323:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  404328:	48 89 c7             	mov    %rax,%rdi
  40432b:	45 31 f6             	xor    %r14d,%r14d
  40432e:	45 31 ed             	xor    %r13d,%r13d
  404331:	eb 9f                	jmp    4042d2 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x592>
  404333:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  404338:	48 89 c7             	mov    %rax,%rdi
  40433b:	e9 60 fb ff ff       	jmpq   403ea0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x160>
  404340:	e8 cb d1 ff ff       	callq  401510 <_ZSt17__throw_bad_allocv@plt>
  404345:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40434c:	00 00 00 
  40434f:	90                   	nop

0000000000404350 <__libc_csu_init>:
  404350:	41 57                	push   %r15
  404352:	41 89 ff             	mov    %edi,%r15d
  404355:	41 56                	push   %r14
  404357:	49 89 f6             	mov    %rsi,%r14
  40435a:	41 55                	push   %r13
  40435c:	49 89 d5             	mov    %rdx,%r13
  40435f:	41 54                	push   %r12
  404361:	4c 8d 25 50 09 20 00 	lea    0x200950(%rip),%r12        # 604cb8 <__frame_dummy_init_array_entry>
  404368:	55                   	push   %rbp
  404369:	48 8d 2d 58 09 20 00 	lea    0x200958(%rip),%rbp        # 604cc8 <__init_array_end>
  404370:	53                   	push   %rbx
  404371:	4c 29 e5             	sub    %r12,%rbp
  404374:	31 db                	xor    %ebx,%ebx
  404376:	48 c1 fd 03          	sar    $0x3,%rbp
  40437a:	48 83 ec 08          	sub    $0x8,%rsp
  40437e:	e8 25 d1 ff ff       	callq  4014a8 <_init>
  404383:	48 85 ed             	test   %rbp,%rbp
  404386:	74 1e                	je     4043a6 <__libc_csu_init+0x56>
  404388:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40438f:	00 
  404390:	4c 89 ea             	mov    %r13,%rdx
  404393:	4c 89 f6             	mov    %r14,%rsi
  404396:	44 89 ff             	mov    %r15d,%edi
  404399:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40439d:	48 83 c3 01          	add    $0x1,%rbx
  4043a1:	48 39 eb             	cmp    %rbp,%rbx
  4043a4:	75 ea                	jne    404390 <__libc_csu_init+0x40>
  4043a6:	48 83 c4 08          	add    $0x8,%rsp
  4043aa:	5b                   	pop    %rbx
  4043ab:	5d                   	pop    %rbp
  4043ac:	41 5c                	pop    %r12
  4043ae:	41 5d                	pop    %r13
  4043b0:	41 5e                	pop    %r14
  4043b2:	41 5f                	pop    %r15
  4043b4:	c3                   	retq   
  4043b5:	90                   	nop
  4043b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4043bd:	00 00 00 

00000000004043c0 <__libc_csu_fini>:
  4043c0:	f3 c3                	repz retq 

Disassembly of section .fini:

00000000004043c4 <_fini>:
  4043c4:	48 83 ec 08          	sub    $0x8,%rsp
  4043c8:	48 83 c4 08          	add    $0x8,%rsp
  4043cc:	c3                   	retq   
