
./build/parallel:     file format elf64-x86-64


Disassembly of section .init:

0000000000401470 <_init>:
  401470:	48 83 ec 08          	sub    $0x8,%rsp
  401474:	48 8b 05 7d 4b 20 00 	mov    0x204b7d(%rip),%rax        # 605ff8 <__gmon_start__>
  40147b:	48 85 c0             	test   %rax,%rax
  40147e:	74 05                	je     401485 <_init+0x15>
  401480:	e8 ab 00 00 00       	callq  401530 <__gmon_start__@plt>
  401485:	48 83 c4 08          	add    $0x8,%rsp
  401489:	c3                   	retq   

Disassembly of section .plt:

0000000000401490 <.plt>:
  401490:	ff 35 72 4b 20 00    	pushq  0x204b72(%rip)        # 606008 <_GLOBAL_OFFSET_TABLE_+0x8>
  401496:	ff 25 74 4b 20 00    	jmpq   *0x204b74(%rip)        # 606010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40149c:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004014a0 <__cxa_begin_catch@plt>:
  4014a0:	ff 25 72 4b 20 00    	jmpq   *0x204b72(%rip)        # 606018 <__cxa_begin_catch@CXXABI_1.3>
  4014a6:	68 00 00 00 00       	pushq  $0x0
  4014ab:	e9 e0 ff ff ff       	jmpq   401490 <.plt>

00000000004014b0 <memcpy@plt>:
  4014b0:	ff 25 6a 4b 20 00    	jmpq   *0x204b6a(%rip)        # 606020 <memcpy@GLIBC_2.14>
  4014b6:	68 01 00 00 00       	pushq  $0x1
  4014bb:	e9 d0 ff ff ff       	jmpq   401490 <.plt>

00000000004014c0 <_ZN2cv3Mat8copySizeERKS0_@plt>:
  4014c0:	ff 25 62 4b 20 00    	jmpq   *0x204b62(%rip)        # 606028 <_ZN2cv3Mat8copySizeERKS0_>
  4014c6:	68 02 00 00 00       	pushq  $0x2
  4014cb:	e9 c0 ff ff ff       	jmpq   401490 <.plt>

00000000004014d0 <_ZSt17__throw_bad_allocv@plt>:
  4014d0:	ff 25 5a 4b 20 00    	jmpq   *0x204b5a(%rip)        # 606030 <_ZSt17__throw_bad_allocv@GLIBCXX_3.4>
  4014d6:	68 03 00 00 00       	pushq  $0x3
  4014db:	e9 b0 ff ff ff       	jmpq   401490 <.plt>

00000000004014e0 <_Znwm@plt>:
  4014e0:	ff 25 52 4b 20 00    	jmpq   *0x204b52(%rip)        # 606038 <_Znwm@GLIBCXX_3.4>
  4014e6:	68 04 00 00 00       	pushq  $0x4
  4014eb:	e9 a0 ff ff ff       	jmpq   401490 <.plt>

00000000004014f0 <free@plt>:
  4014f0:	ff 25 4a 4b 20 00    	jmpq   *0x204b4a(%rip)        # 606040 <free@GLIBC_2.2.5>
  4014f6:	68 05 00 00 00       	pushq  $0x5
  4014fb:	e9 90 ff ff ff       	jmpq   401490 <.plt>

0000000000401500 <_ZN2cv12_OutputArrayC1ERNS_3MatE@plt>:
  401500:	ff 25 42 4b 20 00    	jmpq   *0x204b42(%rip)        # 606048 <_ZN2cv12_OutputArrayC1ERNS_3MatE>
  401506:	68 06 00 00 00       	pushq  $0x6
  40150b:	e9 80 ff ff ff       	jmpq   401490 <.plt>

0000000000401510 <_ZNSolsEi@plt>:
  401510:	ff 25 3a 4b 20 00    	jmpq   *0x204b3a(%rip)        # 606050 <_ZNSolsEi@GLIBCXX_3.4>
  401516:	68 07 00 00 00       	pushq  $0x7
  40151b:	e9 70 ff ff ff       	jmpq   401490 <.plt>

0000000000401520 <_ZSt16__throw_bad_castv@plt>:
  401520:	ff 25 32 4b 20 00    	jmpq   *0x204b32(%rip)        # 606058 <_ZSt16__throw_bad_castv@GLIBCXX_3.4>
  401526:	68 08 00 00 00       	pushq  $0x8
  40152b:	e9 60 ff ff ff       	jmpq   401490 <.plt>

0000000000401530 <__gmon_start__@plt>:
  401530:	ff 25 2a 4b 20 00    	jmpq   *0x204b2a(%rip)        # 606060 <__gmon_start__>
  401536:	68 09 00 00 00       	pushq  $0x9
  40153b:	e9 50 ff ff ff       	jmpq   401490 <.plt>

0000000000401540 <omp_get_num_threads@plt>:
  401540:	ff 25 22 4b 20 00    	jmpq   *0x204b22(%rip)        # 606068 <omp_get_num_threads@OMP_1.0>
  401546:	68 0a 00 00 00       	pushq  $0xa
  40154b:	e9 40 ff ff ff       	jmpq   401490 <.plt>

0000000000401550 <omp_get_thread_num@plt>:
  401550:	ff 25 1a 4b 20 00    	jmpq   *0x204b1a(%rip)        # 606070 <omp_get_thread_num@OMP_1.0>
  401556:	68 0b 00 00 00       	pushq  $0xb
  40155b:	e9 30 ff ff ff       	jmpq   401490 <.plt>

0000000000401560 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>:
  401560:	ff 25 12 4b 20 00    	jmpq   *0x204b12(%rip)        # 606078 <_ZNKSt5ctypeIcE13_M_widen_initEv@GLIBCXX_3.4.11>
  401566:	68 0c 00 00 00       	pushq  $0xc
  40156b:	e9 20 ff ff ff       	jmpq   401490 <.plt>

0000000000401570 <_ZNSo5flushEv@plt>:
  401570:	ff 25 0a 4b 20 00    	jmpq   *0x204b0a(%rip)        # 606080 <_ZNSo5flushEv@GLIBCXX_3.4>
  401576:	68 0d 00 00 00       	pushq  $0xd
  40157b:	e9 10 ff ff ff       	jmpq   401490 <.plt>

0000000000401580 <_ZN2cv3Mat10deallocateEv@plt>:
  401580:	ff 25 02 4b 20 00    	jmpq   *0x204b02(%rip)        # 606088 <_ZN2cv3Mat10deallocateEv>
  401586:	68 0e 00 00 00       	pushq  $0xe
  40158b:	e9 00 ff ff ff       	jmpq   401490 <.plt>

0000000000401590 <_ZdlPv@plt>:
  401590:	ff 25 fa 4a 20 00    	jmpq   *0x204afa(%rip)        # 606090 <_ZdlPv@GLIBCXX_3.4>
  401596:	68 0f 00 00 00       	pushq  $0xf
  40159b:	e9 f0 fe ff ff       	jmpq   401490 <.plt>

00000000004015a0 <_ZNSo3putEc@plt>:
  4015a0:	ff 25 f2 4a 20 00    	jmpq   *0x204af2(%rip)        # 606098 <_ZNSo3putEc@GLIBCXX_3.4>
  4015a6:	68 10 00 00 00       	pushq  $0x10
  4015ab:	e9 e0 fe ff ff       	jmpq   401490 <.plt>

00000000004015b0 <__gxx_personality_v0@plt>:
  4015b0:	ff 25 ea 4a 20 00    	jmpq   *0x204aea(%rip)        # 6060a0 <__gxx_personality_v0@CXXABI_1.3>
  4015b6:	68 11 00 00 00       	pushq  $0x11
  4015bb:	e9 d0 fe ff ff       	jmpq   401490 <.plt>

00000000004015c0 <_ZN2cv7imwriteERKSsRKNS_11_InputArrayERKSt6vectorIiSaIiEE@plt>:
  4015c0:	ff 25 e2 4a 20 00    	jmpq   *0x204ae2(%rip)        # 6060a8 <_ZN2cv7imwriteERKSsRKNS_11_InputArrayERKSt6vectorIiSaIiEE>
  4015c6:	68 12 00 00 00       	pushq  $0x12
  4015cb:	e9 c0 fe ff ff       	jmpq   401490 <.plt>

00000000004015d0 <GOMP_parallel_end@plt>:
  4015d0:	ff 25 da 4a 20 00    	jmpq   *0x204ada(%rip)        # 6060b0 <GOMP_parallel_end@GOMP_1.0>
  4015d6:	68 13 00 00 00       	pushq  $0x13
  4015db:	e9 b0 fe ff ff       	jmpq   401490 <.plt>

00000000004015e0 <_Unwind_Resume@plt>:
  4015e0:	ff 25 d2 4a 20 00    	jmpq   *0x204ad2(%rip)        # 6060b8 <_Unwind_Resume@GCC_3.0>
  4015e6:	68 14 00 00 00       	pushq  $0x14
  4015eb:	e9 a0 fe ff ff       	jmpq   401490 <.plt>

00000000004015f0 <_ZNSt8ios_base4InitD1Ev@plt>:
  4015f0:	ff 25 ca 4a 20 00    	jmpq   *0x204aca(%rip)        # 6060c0 <_ZNSt8ios_base4InitD1Ev@GLIBCXX_3.4>
  4015f6:	68 15 00 00 00       	pushq  $0x15
  4015fb:	e9 90 fe ff ff       	jmpq   401490 <.plt>

0000000000401600 <_ZN2cv11_InputArrayC1ERKNS_3MatE@plt>:
  401600:	ff 25 c2 4a 20 00    	jmpq   *0x204ac2(%rip)        # 6060c8 <_ZN2cv11_InputArrayC1ERKNS_3MatE>
  401606:	68 16 00 00 00       	pushq  $0x16
  40160b:	e9 80 fe ff ff       	jmpq   401490 <.plt>

0000000000401610 <_ZN2cv5mergeERKSt6vectorINS_3MatESaIS1_EERKNS_12_OutputArrayE@plt>:
  401610:	ff 25 ba 4a 20 00    	jmpq   *0x204aba(%rip)        # 6060d0 <_ZN2cv5mergeERKSt6vectorINS_3MatESaIS1_EERKNS_12_OutputArrayE>
  401616:	68 17 00 00 00       	pushq  $0x17
  40161b:	e9 70 fe ff ff       	jmpq   401490 <.plt>

0000000000401620 <_ZNSo9_M_insertIdEERSoT_@plt>:
  401620:	ff 25 b2 4a 20 00    	jmpq   *0x204ab2(%rip)        # 6060d8 <_ZNSo9_M_insertIdEERSoT_@GLIBCXX_3.4.9>
  401626:	68 18 00 00 00       	pushq  $0x18
  40162b:	e9 60 fe ff ff       	jmpq   401490 <.plt>

0000000000401630 <__cxa_end_catch@plt>:
  401630:	ff 25 aa 4a 20 00    	jmpq   *0x204aaa(%rip)        # 6060e0 <__cxa_end_catch@CXXABI_1.3>
  401636:	68 19 00 00 00       	pushq  $0x19
  40163b:	e9 50 fe ff ff       	jmpq   401490 <.plt>

0000000000401640 <_ZNSs4_Rep10_M_disposeERKSaIcE@plt>:
  401640:	ff 25 a2 4a 20 00    	jmpq   *0x204aa2(%rip)        # 6060e8 <_ZNSs4_Rep10_M_disposeERKSaIcE@GLIBCXX_3.4>
  401646:	68 1a 00 00 00       	pushq  $0x1a
  40164b:	e9 40 fe ff ff       	jmpq   401490 <.plt>

0000000000401650 <__cxa_rethrow@plt>:
  401650:	ff 25 9a 4a 20 00    	jmpq   *0x204a9a(%rip)        # 6060f0 <__cxa_rethrow@CXXABI_1.3>
  401656:	68 1b 00 00 00       	pushq  $0x1b
  40165b:	e9 30 fe ff ff       	jmpq   401490 <.plt>

0000000000401660 <calloc@plt>:
  401660:	ff 25 92 4a 20 00    	jmpq   *0x204a92(%rip)        # 6060f8 <calloc@GLIBC_2.2.5>
  401666:	68 1c 00 00 00       	pushq  $0x1c
  40166b:	e9 20 fe ff ff       	jmpq   401490 <.plt>

0000000000401670 <__cxa_atexit@plt>:
  401670:	ff 25 8a 4a 20 00    	jmpq   *0x204a8a(%rip)        # 606100 <__cxa_atexit@GLIBC_2.2.5>
  401676:	68 1d 00 00 00       	pushq  $0x1d
  40167b:	e9 10 fe ff ff       	jmpq   401490 <.plt>

0000000000401680 <omp_set_num_threads@plt>:
  401680:	ff 25 82 4a 20 00    	jmpq   *0x204a82(%rip)        # 606108 <omp_set_num_threads@OMP_1.0>
  401686:	68 1e 00 00 00       	pushq  $0x1e
  40168b:	e9 00 fe ff ff       	jmpq   401490 <.plt>

0000000000401690 <_ZN2cv8fastFreeEPv@plt>:
  401690:	ff 25 7a 4a 20 00    	jmpq   *0x204a7a(%rip)        # 606110 <_ZN2cv8fastFreeEPv>
  401696:	68 1f 00 00 00       	pushq  $0x1f
  40169b:	e9 f0 fd ff ff       	jmpq   401490 <.plt>

00000000004016a0 <__pthread_key_create@plt>:
  4016a0:	ff 25 72 4a 20 00    	jmpq   *0x204a72(%rip)        # 606118 <__pthread_key_create@GLIBC_2.2.5>
  4016a6:	68 20 00 00 00       	pushq  $0x20
  4016ab:	e9 e0 fd ff ff       	jmpq   401490 <.plt>

00000000004016b0 <__libc_start_main@plt>:
  4016b0:	ff 25 6a 4a 20 00    	jmpq   *0x204a6a(%rip)        # 606120 <__libc_start_main@GLIBC_2.2.5>
  4016b6:	68 21 00 00 00       	pushq  $0x21
  4016bb:	e9 d0 fd ff ff       	jmpq   401490 <.plt>

00000000004016c0 <_ZNSs4_Rep10_M_destroyERKSaIcE@plt>:
  4016c0:	ff 25 62 4a 20 00    	jmpq   *0x204a62(%rip)        # 606128 <_ZNSs4_Rep10_M_destroyERKSaIcE@GLIBCXX_3.4>
  4016c6:	68 22 00 00 00       	pushq  $0x22
  4016cb:	e9 c0 fd ff ff       	jmpq   401490 <.plt>

00000000004016d0 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>:
  4016d0:	ff 25 5a 4a 20 00    	jmpq   *0x204a5a(%rip)        # 606130 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@GLIBCXX_3.4.9>
  4016d6:	68 23 00 00 00       	pushq  $0x23
  4016db:	e9 b0 fd ff ff       	jmpq   401490 <.plt>

00000000004016e0 <_ZNSt8ios_base4InitC1Ev@plt>:
  4016e0:	ff 25 52 4a 20 00    	jmpq   *0x204a52(%rip)        # 606138 <_ZNSt8ios_base4InitC1Ev@GLIBCXX_3.4>
  4016e6:	68 24 00 00 00       	pushq  $0x24
  4016eb:	e9 a0 fd ff ff       	jmpq   401490 <.plt>

00000000004016f0 <_ZN2cv5splitERKNS_3MatEPS0_@plt>:
  4016f0:	ff 25 4a 4a 20 00    	jmpq   *0x204a4a(%rip)        # 606140 <_ZN2cv5splitERKNS_3MatEPS0_>
  4016f6:	68 25 00 00 00       	pushq  $0x25
  4016fb:	e9 90 fd ff ff       	jmpq   401490 <.plt>

0000000000401700 <_ZN2cv6imreadERKSsi@plt>:
  401700:	ff 25 42 4a 20 00    	jmpq   *0x204a42(%rip)        # 606148 <_ZN2cv6imreadERKSsi>
  401706:	68 26 00 00 00       	pushq  $0x26
  40170b:	e9 80 fd ff ff       	jmpq   401490 <.plt>

0000000000401710 <_ZNSsC1EPKcRKSaIcE@plt>:
  401710:	ff 25 3a 4a 20 00    	jmpq   *0x204a3a(%rip)        # 606150 <_ZNSsC1EPKcRKSaIcE@GLIBCXX_3.4>
  401716:	68 27 00 00 00       	pushq  $0x27
  40171b:	e9 70 fd ff ff       	jmpq   401490 <.plt>

0000000000401720 <GOMP_parallel_start@plt>:
  401720:	ff 25 32 4a 20 00    	jmpq   *0x204a32(%rip)        # 606158 <GOMP_parallel_start@GOMP_1.0>
  401726:	68 28 00 00 00       	pushq  $0x28
  40172b:	e9 60 fd ff ff       	jmpq   401490 <.plt>

Disassembly of section .text:

0000000000401730 <main>:
  401730:	55                   	push   %rbp
  401731:	be 04 00 00 00       	mov    $0x4,%esi
  401736:	bf 00 10 00 00       	mov    $0x1000,%edi
  40173b:	48 89 e5             	mov    %rsp,%rbp
  40173e:	41 57                	push   %r15
  401740:	41 56                	push   %r14
  401742:	41 55                	push   %r13
  401744:	41 bd 40 00 00 00    	mov    $0x40,%r13d
  40174a:	41 54                	push   %r12
  40174c:	41 bc 40 00 00 00    	mov    $0x40,%r12d
  401752:	53                   	push   %rbx
  401753:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  401757:	48 81 ec c0 00 00 00 	sub    $0xc0,%rsp
  40175e:	c5 fb 11 5c 24 10    	vmovsd %xmm3,0x10(%rsp)
  401764:	c5 fb 11 54 24 18    	vmovsd %xmm2,0x18(%rsp)
  40176a:	e8 f1 fe ff ff       	callq  401660 <calloc@plt>
  40176f:	be 04 00 00 00       	mov    $0x4,%esi
  401774:	bf 00 10 00 00       	mov    $0x1000,%edi
  401779:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40177e:	e8 dd fe ff ff       	callq  401660 <calloc@plt>
  401783:	be 04 00 00 00       	mov    $0x4,%esi
  401788:	bf 00 10 00 00       	mov    $0x1000,%edi
  40178d:	49 89 c7             	mov    %rax,%r15
  401790:	e8 cb fe ff ff       	callq  401660 <calloc@plt>
  401795:	be 04 00 00 00       	mov    $0x4,%esi
  40179a:	bf 00 04 00 00       	mov    $0x400,%edi
  40179f:	49 89 c6             	mov    %rax,%r14
  4017a2:	e8 b9 fe ff ff       	callq  401660 <calloc@plt>
  4017a7:	be 04 00 00 00       	mov    $0x4,%esi
  4017ac:	bf 00 04 00 00       	mov    $0x400,%edi
  4017b1:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  4017b6:	e8 a5 fe ff ff       	callq  401660 <calloc@plt>
  4017bb:	be 04 00 00 00       	mov    $0x4,%esi
  4017c0:	bf 00 04 00 00       	mov    $0x400,%edi
  4017c5:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  4017ca:	e8 91 fe ff ff       	callq  401660 <calloc@plt>
  4017cf:	4c 8b 54 24 48       	mov    0x48(%rsp),%r10
  4017d4:	4c 8b 4c 24 40       	mov    0x40(%rsp),%r9
  4017d9:	48 89 c2             	mov    %rax,%rdx
  4017dc:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  4017e1:	4c 89 d6             	mov    %r10,%rsi
  4017e4:	4c 89 cf             	mov    %r9,%rdi
  4017e7:	4c 89 54 24 30       	mov    %r10,0x30(%rsp)
  4017ec:	4c 89 4c 24 38       	mov    %r9,0x38(%rsp)
  4017f1:	e8 9a 18 00 00       	callq  403090 <_Z11decodeImagePfS_S_>
  4017f6:	be 04 00 00 00       	mov    $0x4,%esi
  4017fb:	bf 40 00 00 00       	mov    $0x40,%edi
  401800:	e8 5b fe ff ff       	callq  401660 <calloc@plt>
  401805:	48 89 c7             	mov    %rax,%rdi
  401808:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  40180d:	e8 6e 09 00 00       	callq  402180 <_Z20generateCoefficientsPf>
  401812:	c5 f1 57 c9          	vxorpd %xmm1,%xmm1,%xmm1
  401816:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
  40181b:	4c 8b 5c 24 28       	mov    0x28(%rsp),%r11
  401820:	4c 8b 54 24 30       	mov    0x30(%rsp),%r10
  401825:	4c 8b 4c 24 38       	mov    0x38(%rsp),%r9
  40182a:	4c 8b 44 24 40       	mov    0x40(%rsp),%r8
  40182f:	c5 fb 10 5c 24 10    	vmovsd 0x10(%rsp),%xmm3
  401835:	c7 44 24 48 01 00 00 	movl   $0x1,0x48(%rsp)
  40183c:	00 
  40183d:	c5 fb 10 54 24 18    	vmovsd 0x18(%rsp),%xmm2
  401843:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  401848:	8b 7c 24 48          	mov    0x48(%rsp),%edi
  40184c:	4c 89 44 24 08       	mov    %r8,0x8(%rsp)
  401851:	c5 fb 11 5c 24 30    	vmovsd %xmm3,0x30(%rsp)
  401857:	4c 89 5c 24 10       	mov    %r11,0x10(%rsp)
  40185c:	4c 89 54 24 18       	mov    %r10,0x18(%rsp)
  401861:	c5 fb 11 54 24 38    	vmovsd %xmm2,0x38(%rsp)
  401867:	4c 89 4c 24 20       	mov    %r9,0x20(%rsp)
  40186c:	c5 fb 11 4c 24 40    	vmovsd %xmm1,0x40(%rsp)
  401872:	bb 64 00 00 00       	mov    $0x64,%ebx
  401877:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  40187c:	e8 ff fd ff ff       	callq  401680 <omp_set_num_threads@plt>
  401881:	4c 8b 44 24 08       	mov    0x8(%rsp),%r8
  401886:	4c 8b 5c 24 10       	mov    0x10(%rsp),%r11
  40188b:	4c 8b 54 24 18       	mov    0x18(%rsp),%r10
  401890:	4c 8b 4c 24 20       	mov    0x20(%rsp),%r9
  401895:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
  40189a:	c5 fb 10 5c 24 30    	vmovsd 0x30(%rsp),%xmm3
  4018a0:	c5 fb 10 54 24 38    	vmovsd 0x38(%rsp),%xmm2
  4018a6:	c5 fb 10 4c 24 40    	vmovsd 0x40(%rsp),%xmm1
  4018ac:	0f 1f 40 00          	nopl   0x0(%rax)
  4018b0:	48 8d 74 24 60       	lea    0x60(%rsp),%rsi
  4018b5:	31 d2                	xor    %edx,%edx
  4018b7:	bf 00 1d 40 00       	mov    $0x401d00,%edi
  4018bc:	c5 fb 11 4c 24 60    	vmovsd %xmm1,0x60(%rsp)
  4018c2:	48 89 4c 24 78       	mov    %rcx,0x78(%rsp)
  4018c7:	4c 89 8c 24 90 00 00 	mov    %r9,0x90(%rsp)
  4018ce:	00 
  4018cf:	c5 fb 11 54 24 68    	vmovsd %xmm2,0x68(%rsp)
  4018d5:	4c 89 94 24 98 00 00 	mov    %r10,0x98(%rsp)
  4018dc:	00 
  4018dd:	4c 89 9c 24 a0 00 00 	mov    %r11,0xa0(%rsp)
  4018e4:	00 
  4018e5:	c5 fb 11 5c 24 70    	vmovsd %xmm3,0x70(%rsp)
  4018eb:	4c 89 84 24 a8 00 00 	mov    %r8,0xa8(%rsp)
  4018f2:	00 
  4018f3:	4c 89 bc 24 80 00 00 	mov    %r15,0x80(%rsp)
  4018fa:	00 
  4018fb:	4c 89 b4 24 88 00 00 	mov    %r14,0x88(%rsp)
  401902:	00 
  401903:	44 89 a4 24 b0 00 00 	mov    %r12d,0xb0(%rsp)
  40190a:	00 
  40190b:	44 89 ac 24 b4 00 00 	mov    %r13d,0xb4(%rsp)
  401912:	00 
  401913:	e8 08 fe ff ff       	callq  401720 <GOMP_parallel_start@plt>
  401918:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  40191d:	e8 de 03 00 00       	callq  401d00 <main._omp_fn.0>
  401922:	e8 a9 fc ff ff       	callq  4015d0 <GOMP_parallel_end@plt>
  401927:	83 eb 01             	sub    $0x1,%ebx
  40192a:	c5 fb 10 4c 24 60    	vmovsd 0x60(%rsp),%xmm1
  401930:	c5 fb 10 54 24 68    	vmovsd 0x68(%rsp),%xmm2
  401936:	48 8b 4c 24 78       	mov    0x78(%rsp),%rcx
  40193b:	c5 fb 10 5c 24 70    	vmovsd 0x70(%rsp),%xmm3
  401941:	4c 8b bc 24 80 00 00 	mov    0x80(%rsp),%r15
  401948:	00 
  401949:	4c 8b b4 24 88 00 00 	mov    0x88(%rsp),%r14
  401950:	00 
  401951:	4c 8b 8c 24 90 00 00 	mov    0x90(%rsp),%r9
  401958:	00 
  401959:	4c 8b 94 24 98 00 00 	mov    0x98(%rsp),%r10
  401960:	00 
  401961:	4c 8b 9c 24 a0 00 00 	mov    0xa0(%rsp),%r11
  401968:	00 
  401969:	4c 8b 84 24 a8 00 00 	mov    0xa8(%rsp),%r8
  401970:	00 
  401971:	44 8b a4 24 b0 00 00 	mov    0xb0(%rsp),%r12d
  401978:	00 
  401979:	44 8b ac 24 b4 00 00 	mov    0xb4(%rsp),%r13d
  401980:	00 
  401981:	0f 85 29 ff ff ff    	jne    4018b0 <main+0x180>
  401987:	c5 f3 59 0d 41 32 00 	vmulsd 0x3241(%rip),%xmm1,%xmm1        # 404bd0 <__dso_handle+0xd8>
  40198e:	00 
  40198f:	44 89 e0             	mov    %r12d,%eax
  401992:	8b 74 24 48          	mov    0x48(%rsp),%esi
  401996:	41 0f af c5          	imul   %r13d,%eax
  40199a:	bf a0 61 60 00       	mov    $0x6061a0,%edi
  40199f:	4c 89 04 24          	mov    %r8,(%rsp)
  4019a3:	4c 89 5c 24 08       	mov    %r11,0x8(%rsp)
  4019a8:	4c 89 54 24 10       	mov    %r10,0x10(%rsp)
  4019ad:	c5 fb 11 5c 24 28    	vmovsd %xmm3,0x28(%rsp)
  4019b3:	4c 89 4c 24 18       	mov    %r9,0x18(%rsp)
  4019b8:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
  4019bd:	c5 fb 11 54 24 30    	vmovsd %xmm2,0x30(%rsp)
  4019c3:	8d 50 0f             	lea    0xf(%rax),%edx
  4019c6:	85 c0                	test   %eax,%eax
  4019c8:	c5 f3 5e 0d 08 32 00 	vdivsd 0x3208(%rip),%xmm1,%xmm1        # 404bd8 <__dso_handle+0xe0>
  4019cf:	00 
  4019d0:	0f 48 c2             	cmovs  %edx,%eax
  4019d3:	c1 f8 04             	sar    $0x4,%eax
  4019d6:	8d 04 40             	lea    (%rax,%rax,2),%eax
  4019d9:	c1 e0 07             	shl    $0x7,%eax
  4019dc:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  4019e0:	c5 f3 5e 0d f8 31 00 	vdivsd 0x31f8(%rip),%xmm1,%xmm1        # 404be0 <__dso_handle+0xe8>
  4019e7:	00 
  4019e8:	c5 fb 5e c1          	vdivsd %xmm1,%xmm0,%xmm0
  4019ec:	c5 fb 11 4c 24 38    	vmovsd %xmm1,0x38(%rsp)
  4019f2:	c5 fb 11 44 24 40    	vmovsd %xmm0,0x40(%rsp)
  4019f8:	e8 13 fb ff ff       	callq  401510 <_ZNSolsEi@plt>
  4019fd:	ba 01 00 00 00       	mov    $0x1,%edx
  401a02:	be 66 4b 40 00       	mov    $0x404b66,%esi
  401a07:	48 89 c7             	mov    %rax,%rdi
  401a0a:	48 89 c3             	mov    %rax,%rbx
  401a0d:	e8 be fc ff ff       	callq  4016d0 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
  401a12:	c5 fb 10 44 24 40    	vmovsd 0x40(%rsp),%xmm0
  401a18:	48 89 df             	mov    %rbx,%rdi
  401a1b:	e8 00 fc ff ff       	callq  401620 <_ZNSo9_M_insertIdEERSoT_@plt>
  401a20:	48 8d 74 24 5f       	lea    0x5f(%rsp),%rsi
  401a25:	ba 01 00 00 00       	mov    $0x1,%edx
  401a2a:	48 89 c7             	mov    %rax,%rdi
  401a2d:	c6 44 24 5f 2c       	movb   $0x2c,0x5f(%rsp)
  401a32:	e8 99 fc ff ff       	callq  4016d0 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
  401a37:	c5 fb 10 4c 24 38    	vmovsd 0x38(%rsp),%xmm1
  401a3d:	48 89 c7             	mov    %rax,%rdi
  401a40:	c5 f9 28 c1          	vmovapd %xmm1,%xmm0
  401a44:	c5 fb 11 4c 24 40    	vmovsd %xmm1,0x40(%rsp)
  401a4a:	e8 d1 fb ff ff       	callq  401620 <_ZNSo9_M_insertIdEERSoT_@plt>
  401a4f:	48 89 c2             	mov    %rax,%rdx
  401a52:	48 8b 00             	mov    (%rax),%rax
  401a55:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
  401a5a:	c5 fb 10 4c 24 40    	vmovsd 0x40(%rsp),%xmm1
  401a60:	4c 8b 4c 24 18       	mov    0x18(%rsp),%r9
  401a65:	c5 fb 10 54 24 30    	vmovsd 0x30(%rsp),%xmm2
  401a6b:	4c 8b 54 24 10       	mov    0x10(%rsp),%r10
  401a70:	48 8b 40 e8          	mov    -0x18(%rax),%rax
  401a74:	c5 fb 10 5c 24 28    	vmovsd 0x28(%rsp),%xmm3
  401a7a:	4c 8b 5c 24 08       	mov    0x8(%rsp),%r11
  401a7f:	4c 8b 04 24          	mov    (%rsp),%r8
  401a83:	48 8b 9c 02 f0 00 00 	mov    0xf0(%rdx,%rax,1),%rbx
  401a8a:	00 
  401a8b:	48 85 db             	test   %rbx,%rbx
  401a8e:	0f 84 53 01 00 00    	je     401be7 <main+0x4b7>
  401a94:	80 7b 38 00          	cmpb   $0x0,0x38(%rbx)
  401a98:	0f 84 d0 00 00 00    	je     401b6e <main+0x43e>
  401a9e:	0f b6 43 43          	movzbl 0x43(%rbx),%eax
  401aa2:	48 89 d7             	mov    %rdx,%rdi
  401aa5:	0f be f0             	movsbl %al,%esi
  401aa8:	4c 89 44 24 08       	mov    %r8,0x8(%rsp)
  401aad:	4c 89 5c 24 10       	mov    %r11,0x10(%rsp)
  401ab2:	4c 89 54 24 18       	mov    %r10,0x18(%rsp)
  401ab7:	c5 fb 11 5c 24 30    	vmovsd %xmm3,0x30(%rsp)
  401abd:	4c 89 4c 24 20       	mov    %r9,0x20(%rsp)
  401ac2:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  401ac7:	c5 fb 11 54 24 38    	vmovsd %xmm2,0x38(%rsp)
  401acd:	c5 fb 11 4c 24 40    	vmovsd %xmm1,0x40(%rsp)
  401ad3:	e8 c8 fa ff ff       	callq  4015a0 <_ZNSo3putEc@plt>
  401ad8:	48 89 c7             	mov    %rax,%rdi
  401adb:	e8 90 fa ff ff       	callq  401570 <_ZNSo5flushEv@plt>
  401ae0:	83 44 24 48 01       	addl   $0x1,0x48(%rsp)
  401ae5:	c5 fb 10 4c 24 40    	vmovsd 0x40(%rsp),%xmm1
  401aeb:	83 7c 24 48 19       	cmpl   $0x19,0x48(%rsp)
  401af0:	c5 fb 10 54 24 38    	vmovsd 0x38(%rsp),%xmm2
  401af6:	c5 fb 10 5c 24 30    	vmovsd 0x30(%rsp),%xmm3
  401afc:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
  401b01:	4c 8b 4c 24 20       	mov    0x20(%rsp),%r9
  401b06:	4c 8b 54 24 18       	mov    0x18(%rsp),%r10
  401b0b:	4c 8b 5c 24 10       	mov    0x10(%rsp),%r11
  401b10:	4c 8b 44 24 08       	mov    0x8(%rsp),%r8
  401b15:	0f 85 2d fd ff ff    	jne    401848 <main+0x118>
  401b1b:	4c 89 f2             	mov    %r14,%rdx
  401b1e:	4c 89 fe             	mov    %r15,%rsi
  401b21:	48 89 cf             	mov    %rcx,%rdi
  401b24:	48 89 4c 24 48       	mov    %rcx,0x48(%rsp)
  401b29:	4c 89 44 24 40       	mov    %r8,0x40(%rsp)
  401b2e:	e8 7d 0c 00 00       	callq  4027b0 <_Z11encodeImagePfS_S_>
  401b33:	4c 8b 44 24 40       	mov    0x40(%rsp),%r8
  401b38:	4c 89 c7             	mov    %r8,%rdi
  401b3b:	e8 b0 f9 ff ff       	callq  4014f0 <free@plt>
  401b40:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  401b45:	48 89 cf             	mov    %rcx,%rdi
  401b48:	e8 a3 f9 ff ff       	callq  4014f0 <free@plt>
  401b4d:	4c 89 ff             	mov    %r15,%rdi
  401b50:	e8 9b f9 ff ff       	callq  4014f0 <free@plt>
  401b55:	4c 89 f7             	mov    %r14,%rdi
  401b58:	e8 93 f9 ff ff       	callq  4014f0 <free@plt>
  401b5d:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
  401b61:	31 c0                	xor    %eax,%eax
  401b63:	5b                   	pop    %rbx
  401b64:	41 5c                	pop    %r12
  401b66:	41 5d                	pop    %r13
  401b68:	41 5e                	pop    %r14
  401b6a:	41 5f                	pop    %r15
  401b6c:	5d                   	pop    %rbp
  401b6d:	c3                   	retq   
  401b6e:	48 89 df             	mov    %rbx,%rdi
  401b71:	48 89 14 24          	mov    %rdx,(%rsp)
  401b75:	4c 89 44 24 08       	mov    %r8,0x8(%rsp)
  401b7a:	4c 89 5c 24 10       	mov    %r11,0x10(%rsp)
  401b7f:	4c 89 54 24 18       	mov    %r10,0x18(%rsp)
  401b84:	c5 fb 11 5c 24 30    	vmovsd %xmm3,0x30(%rsp)
  401b8a:	4c 89 4c 24 20       	mov    %r9,0x20(%rsp)
  401b8f:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  401b94:	c5 fb 11 54 24 38    	vmovsd %xmm2,0x38(%rsp)
  401b9a:	c5 fb 11 4c 24 40    	vmovsd %xmm1,0x40(%rsp)
  401ba0:	e8 bb f9 ff ff       	callq  401560 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>
  401ba5:	48 8b 03             	mov    (%rbx),%rax
  401ba8:	be 0a 00 00 00       	mov    $0xa,%esi
  401bad:	48 89 df             	mov    %rbx,%rdi
  401bb0:	ff 50 30             	callq  *0x30(%rax)
  401bb3:	48 8b 14 24          	mov    (%rsp),%rdx
  401bb7:	4c 8b 44 24 08       	mov    0x8(%rsp),%r8
  401bbc:	4c 8b 5c 24 10       	mov    0x10(%rsp),%r11
  401bc1:	4c 8b 54 24 18       	mov    0x18(%rsp),%r10
  401bc6:	4c 8b 4c 24 20       	mov    0x20(%rsp),%r9
  401bcb:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
  401bd0:	c5 fb 10 5c 24 30    	vmovsd 0x30(%rsp),%xmm3
  401bd6:	c5 fb 10 54 24 38    	vmovsd 0x38(%rsp),%xmm2
  401bdc:	c5 fb 10 4c 24 40    	vmovsd 0x40(%rsp),%xmm1
  401be2:	e9 bb fe ff ff       	jmpq   401aa2 <main+0x372>
  401be7:	e8 34 f9 ff ff       	callq  401520 <_ZSt16__throw_bad_castv@plt>
  401bec:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401bf0 <_GLOBAL__sub_I__Z11decodeImagePfS_S_>:
  401bf0:	48 83 ec 08          	sub    $0x8,%rsp
  401bf4:	bf b1 62 60 00       	mov    $0x6062b1,%edi
  401bf9:	e8 e2 fa ff ff       	callq  4016e0 <_ZNSt8ios_base4InitC1Ev@plt>
  401bfe:	ba f8 4a 40 00       	mov    $0x404af8,%edx
  401c03:	be b1 62 60 00       	mov    $0x6062b1,%esi
  401c08:	bf f0 15 40 00       	mov    $0x4015f0,%edi
  401c0d:	48 83 c4 08          	add    $0x8,%rsp
  401c11:	e9 5a fa ff ff       	jmpq   401670 <__cxa_atexit@plt>

0000000000401c16 <_start>:
  401c16:	31 ed                	xor    %ebp,%ebp
  401c18:	49 89 d1             	mov    %rdx,%r9
  401c1b:	5e                   	pop    %rsi
  401c1c:	48 89 e2             	mov    %rsp,%rdx
  401c1f:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  401c23:	50                   	push   %rax
  401c24:	54                   	push   %rsp
  401c25:	49 c7 c0 e0 4a 40 00 	mov    $0x404ae0,%r8
  401c2c:	48 c7 c1 70 4a 40 00 	mov    $0x404a70,%rcx
  401c33:	48 c7 c7 30 17 40 00 	mov    $0x401730,%rdi
  401c3a:	e8 71 fa ff ff       	callq  4016b0 <__libc_start_main@plt>
  401c3f:	f4                   	hlt    

0000000000401c40 <deregister_tm_clones>:
  401c40:	b8 6f 61 60 00       	mov    $0x60616f,%eax
  401c45:	55                   	push   %rbp
  401c46:	48 2d 68 61 60 00    	sub    $0x606168,%rax
  401c4c:	48 83 f8 0e          	cmp    $0xe,%rax
  401c50:	48 89 e5             	mov    %rsp,%rbp
  401c53:	77 02                	ja     401c57 <deregister_tm_clones+0x17>
  401c55:	5d                   	pop    %rbp
  401c56:	c3                   	retq   
  401c57:	b8 00 00 00 00       	mov    $0x0,%eax
  401c5c:	48 85 c0             	test   %rax,%rax
  401c5f:	74 f4                	je     401c55 <deregister_tm_clones+0x15>
  401c61:	5d                   	pop    %rbp
  401c62:	bf 68 61 60 00       	mov    $0x606168,%edi
  401c67:	ff e0                	jmpq   *%rax
  401c69:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401c70 <register_tm_clones>:
  401c70:	b8 68 61 60 00       	mov    $0x606168,%eax
  401c75:	55                   	push   %rbp
  401c76:	48 2d 68 61 60 00    	sub    $0x606168,%rax
  401c7c:	48 c1 f8 03          	sar    $0x3,%rax
  401c80:	48 89 e5             	mov    %rsp,%rbp
  401c83:	48 89 c2             	mov    %rax,%rdx
  401c86:	48 c1 ea 3f          	shr    $0x3f,%rdx
  401c8a:	48 01 d0             	add    %rdx,%rax
  401c8d:	48 d1 f8             	sar    %rax
  401c90:	75 02                	jne    401c94 <register_tm_clones+0x24>
  401c92:	5d                   	pop    %rbp
  401c93:	c3                   	retq   
  401c94:	ba 00 00 00 00       	mov    $0x0,%edx
  401c99:	48 85 d2             	test   %rdx,%rdx
  401c9c:	74 f4                	je     401c92 <register_tm_clones+0x22>
  401c9e:	5d                   	pop    %rbp
  401c9f:	48 89 c6             	mov    %rax,%rsi
  401ca2:	bf 68 61 60 00       	mov    $0x606168,%edi
  401ca7:	ff e2                	jmpq   *%rdx
  401ca9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401cb0 <__do_global_dtors_aux>:
  401cb0:	80 3d f9 45 20 00 00 	cmpb   $0x0,0x2045f9(%rip)        # 6062b0 <completed.6355>
  401cb7:	75 11                	jne    401cca <__do_global_dtors_aux+0x1a>
  401cb9:	55                   	push   %rbp
  401cba:	48 89 e5             	mov    %rsp,%rbp
  401cbd:	e8 7e ff ff ff       	callq  401c40 <deregister_tm_clones>
  401cc2:	5d                   	pop    %rbp
  401cc3:	c6 05 e6 45 20 00 01 	movb   $0x1,0x2045e6(%rip)        # 6062b0 <completed.6355>
  401cca:	f3 c3                	repz retq 
  401ccc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401cd0 <frame_dummy>:
  401cd0:	48 83 3d f8 3f 20 00 	cmpq   $0x0,0x203ff8(%rip)        # 605cd0 <__JCR_END__>
  401cd7:	00 
  401cd8:	74 1e                	je     401cf8 <frame_dummy+0x28>
  401cda:	b8 00 00 00 00       	mov    $0x0,%eax
  401cdf:	48 85 c0             	test   %rax,%rax
  401ce2:	74 14                	je     401cf8 <frame_dummy+0x28>
  401ce4:	55                   	push   %rbp
  401ce5:	bf d0 5c 60 00       	mov    $0x605cd0,%edi
  401cea:	48 89 e5             	mov    %rsp,%rbp
  401ced:	ff d0                	callq  *%rax
  401cef:	5d                   	pop    %rbp
  401cf0:	e9 7b ff ff ff       	jmpq   401c70 <register_tm_clones>
  401cf5:	0f 1f 00             	nopl   (%rax)
  401cf8:	e9 73 ff ff ff       	jmpq   401c70 <register_tm_clones>
  401cfd:	0f 1f 00             	nopl   (%rax)

0000000000401d00 <main._omp_fn.0>:
  401d00:	41 57                	push   %r15
  401d02:	41 56                	push   %r14
  401d04:	41 55                	push   %r13
  401d06:	41 54                	push   %r12
  401d08:	55                   	push   %rbp
  401d09:	53                   	push   %rbx
  401d0a:	48 89 fb             	mov    %rdi,%rbx
  401d0d:	48 83 ec 68          	sub    $0x68,%rsp
  401d11:	8b 47 50             	mov    0x50(%rdi),%eax
  401d14:	89 44 24 20          	mov    %eax,0x20(%rsp)
  401d18:	0f af 47 54          	imul   0x54(%rdi),%eax
  401d1c:	8d 68 0f             	lea    0xf(%rax),%ebp
  401d1f:	85 c0                	test   %eax,%eax
  401d21:	0f 49 e8             	cmovns %eax,%ebp
  401d24:	c1 fd 04             	sar    $0x4,%ebp
  401d27:	e8 14 f8 ff ff       	callq  401540 <omp_get_num_threads@plt>
  401d2c:	41 89 c4             	mov    %eax,%r12d
  401d2f:	e8 1c f8 ff ff       	callq  401550 <omp_get_thread_num@plt>
  401d34:	89 c7                	mov    %eax,%edi
  401d36:	89 04 24             	mov    %eax,(%rsp)
  401d39:	89 e8                	mov    %ebp,%eax
  401d3b:	99                   	cltd   
  401d3c:	41 f7 fc             	idiv   %r12d
  401d3f:	39 d7                	cmp    %edx,%edi
  401d41:	7d 05                	jge    401d48 <main._omp_fn.0+0x48>
  401d43:	83 c0 01             	add    $0x1,%eax
  401d46:	31 d2                	xor    %edx,%edx
  401d48:	8b 0c 24             	mov    (%rsp),%ecx
  401d4b:	0f af c8             	imul   %eax,%ecx
  401d4e:	8d 34 11             	lea    (%rcx,%rdx,1),%esi
  401d51:	01 f0                	add    %esi,%eax
  401d53:	89 74 24 04          	mov    %esi,0x4(%rsp)
  401d57:	39 c6                	cmp    %eax,%esi
  401d59:	89 44 24 24          	mov    %eax,0x24(%rsp)
  401d5d:	0f 8d e5 03 00 00    	jge    402148 <main._omp_fn.0+0x448>
  401d63:	8d 04 36             	lea    (%rsi,%rsi,1),%eax
  401d66:	8b 7c 24 20          	mov    0x20(%rsp),%edi
  401d6a:	48 8b 4b 48          	mov    0x48(%rbx),%rcx
  401d6e:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  401d72:	48 89 5c 24 58       	mov    %rbx,0x58(%rsp)
  401d77:	89 44 24 18          	mov    %eax,0x18(%rsp)
  401d7b:	48 8b 43 28          	mov    0x28(%rbx),%rax
  401d7f:	48 63 f7             	movslq %edi,%rsi
  401d82:	48 c1 e6 02          	shl    $0x2,%rsi
  401d86:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
  401d8b:	48 8b 43 20          	mov    0x20(%rbx),%rax
  401d8f:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
  401d94:	48 8b 43 18          	mov    0x18(%rbx),%rax
  401d98:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  401d9d:	48 8b 43 40          	mov    0x40(%rbx),%rax
  401da1:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
  401da6:	48 8b 43 38          	mov    0x38(%rbx),%rax
  401daa:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  401daf:	48 8b 43 30          	mov    0x30(%rbx),%rax
  401db3:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  401db8:	89 f8                	mov    %edi,%eax
  401dba:	c1 e8 1f             	shr    $0x1f,%eax
  401dbd:	01 f8                	add    %edi,%eax
  401dbf:	89 44 24 1c          	mov    %eax,0x1c(%rsp)
  401dc3:	d1 7c 24 1c          	sarl   0x1c(%rsp)
  401dc7:	48 63 44 24 1c       	movslq 0x1c(%rsp),%rax
  401dcc:	4c 8d 24 85 00 00 00 	lea    0x0(,%rax,4),%r12
  401dd3:	00 
  401dd4:	49 8d 6c 24 04       	lea    0x4(%r12),%rbp
  401dd9:	4d 89 e3             	mov    %r12,%r11
  401ddc:	e9 09 03 00 00       	jmpq   4020ea <main._omp_fn.0+0x3ea>
  401de1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  401de8:	c4 e1 f3 2a ca       	vcvtsi2sd %rdx,%xmm1,%xmm1
  401ded:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
  401df2:	c5 fb 11 48 08       	vmovsd %xmm1,0x8(%rax)
  401df7:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
  401dfc:	4d 63 c9             	movslq %r9d,%r9
  401dff:	48 8b 5c 24 48       	mov    0x48(%rsp),%rbx
  401e04:	49 c1 e1 02          	shl    $0x2,%r9
  401e08:	48 63 d7             	movslq %edi,%rdx
  401e0b:	48 c1 e2 02          	shl    $0x2,%rdx
  401e0f:	c5 f8 28 09          	vmovaps (%rcx),%xmm1
  401e13:	4e 8d 04 08          	lea    (%rax,%r9,1),%r8
  401e17:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
  401e1c:	48 8d 3c 13          	lea    (%rbx,%rdx,1),%rdi
  401e20:	c5 f8 28 61 10       	vmovaps 0x10(%rcx),%xmm4
  401e25:	4c 89 44 24 08       	mov    %r8,0x8(%rsp)
  401e2a:	4e 8d 34 1f          	lea    (%rdi,%r11,1),%r14
  401e2e:	48 8d 1c 2f          	lea    (%rdi,%rbp,1),%rbx
  401e32:	4e 8d 14 08          	lea    (%rax,%r9,1),%r10
  401e36:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
  401e3b:	4c 03 4c 24 28       	add    0x28(%rsp),%r9
  401e40:	c4 e2 79 18 6f 04    	vbroadcastss 0x4(%rdi),%xmm5
  401e46:	48 01 d0             	add    %rdx,%rax
  401e49:	48 03 54 24 40       	add    0x40(%rsp),%rdx
  401e4e:	c4 e2 79 18 10       	vbroadcastss (%rax),%xmm2
  401e53:	c5 78 28 dd          	vmovaps %xmm5,%xmm11
  401e57:	4e 8d 2c 18          	lea    (%rax,%r11,1),%r13
  401e5b:	4c 8d 04 28          	lea    (%rax,%rbp,1),%r8
  401e5f:	c4 62 79 18 07       	vbroadcastss (%rdi),%xmm8
  401e64:	c4 e2 79 98 d1       	vfmadd132ps %xmm1,%xmm0,%xmm2
  401e69:	c4 62 79 98 dc       	vfmadd132ps %xmm4,%xmm0,%xmm11
  401e6e:	4e 8d 3c 1a          	lea    (%rdx,%r11,1),%r15
  401e72:	4c 8d 24 2a          	lea    (%rdx,%rbp,1),%r12
  401e76:	c4 e2 79 18 72 04    	vbroadcastss 0x4(%rdx),%xmm6
  401e7c:	c4 62 79 98 c1       	vfmadd132ps %xmm1,%xmm0,%xmm8
  401e81:	c4 62 79 18 0a       	vbroadcastss (%rdx),%xmm9
  401e86:	c5 78 28 e6          	vmovaps %xmm6,%xmm12
  401e8a:	c4 62 79 98 c9       	vfmadd132ps %xmm1,%xmm0,%xmm9
  401e8f:	c4 e2 79 18 48 04    	vbroadcastss 0x4(%rax),%xmm1
  401e95:	c4 62 79 98 e4       	vfmadd132ps %xmm4,%xmm0,%xmm12
  401e9a:	c5 f8 28 79 40       	vmovaps 0x40(%rcx),%xmm7
  401e9f:	c4 e2 79 98 e1       	vfmadd132ps %xmm1,%xmm0,%xmm4
  401ea4:	c5 78 28 d4          	vmovaps %xmm4,%xmm10
  401ea8:	c5 f8 28 61 50       	vmovaps 0x50(%rcx),%xmm4
  401ead:	c4 62 49 b8 cf       	vfmadd231ps %xmm7,%xmm6,%xmm9
  401eb2:	c4 62 51 b8 c7       	vfmadd231ps %xmm7,%xmm5,%xmm8
  401eb7:	c4 e2 69 98 f9       	vfmadd132ps %xmm1,%xmm2,%xmm7
  401ebc:	c5 f8 28 99 90 00 00 	vmovaps 0x90(%rcx),%xmm3
  401ec3:	00 
  401ec4:	c4 e2 19 98 f4       	vfmadd132ps %xmm4,%xmm12,%xmm6
  401ec9:	c4 e2 21 98 ec       	vfmadd132ps %xmm4,%xmm11,%xmm5
  401ece:	c4 e2 29 98 e1       	vfmadd132ps %xmm1,%xmm10,%xmm4
  401ed3:	c4 42 79 18 27       	vbroadcastss (%r15),%xmm12
  401ed8:	c4 42 79 18 1e       	vbroadcastss (%r14),%xmm11
  401edd:	c4 e2 19 b8 f3       	vfmadd231ps %xmm3,%xmm12,%xmm6
  401ee2:	c5 f8 28 91 80 00 00 	vmovaps 0x80(%rcx),%xmm2
  401ee9:	00 
  401eea:	c4 e2 21 b8 eb       	vfmadd231ps %xmm3,%xmm11,%xmm5
  401eef:	c4 42 79 18 55 00    	vbroadcastss 0x0(%r13),%xmm10
  401ef5:	c4 62 19 b8 ca       	vfmadd231ps %xmm2,%xmm12,%xmm9
  401efa:	c4 62 21 b8 c2       	vfmadd231ps %xmm2,%xmm11,%xmm8
  401eff:	c4 42 79 18 24 24    	vbroadcastss (%r12),%xmm12
  401f05:	c4 e2 29 b8 fa       	vfmadd231ps %xmm2,%xmm10,%xmm7
  401f0a:	c4 e2 29 b8 e3       	vfmadd231ps %xmm3,%xmm10,%xmm4
  401f0f:	c4 62 79 18 1b       	vbroadcastss (%rbx),%xmm11
  401f14:	4c 89 44 24 10       	mov    %r8,0x10(%rsp)
  401f19:	c4 42 79 18 10       	vbroadcastss (%r8),%xmm10
  401f1e:	4c 8b 44 24 08       	mov    0x8(%rsp),%r8
  401f23:	c5 f8 28 91 c0 00 00 	vmovaps 0xc0(%rcx),%xmm2
  401f2a:	00 
  401f2b:	c5 f8 28 89 d0 00 00 	vmovaps 0xd0(%rcx),%xmm1
  401f32:	00 
  401f33:	c4 62 19 b8 ca       	vfmadd231ps %xmm2,%xmm12,%xmm9
  401f38:	c4 62 21 b8 c2       	vfmadd231ps %xmm2,%xmm11,%xmm8
  401f3d:	c4 c2 41 98 d2       	vfmadd132ps %xmm10,%xmm7,%xmm2
  401f42:	c4 41 78 29 09       	vmovaps %xmm9,(%r9)
  401f47:	c4 62 49 98 e1       	vfmadd132ps %xmm1,%xmm6,%xmm12
  401f4c:	c4 62 51 98 d9       	vfmadd132ps %xmm1,%xmm5,%xmm11
  401f51:	49 01 f1             	add    %rsi,%r9
  401f54:	c4 62 59 98 d1       	vfmadd132ps %xmm1,%xmm4,%xmm10
  401f59:	c4 41 78 29 02       	vmovaps %xmm8,(%r10)
  401f5e:	49 01 f2             	add    %rsi,%r10
  401f61:	c4 c1 78 29 10       	vmovaps %xmm2,(%r8)
  401f66:	49 01 f0             	add    %rsi,%r8
  401f69:	c4 41 78 29 21       	vmovaps %xmm12,(%r9)
  401f6e:	c4 41 78 29 1a       	vmovaps %xmm11,(%r10)
  401f73:	c4 41 78 29 10       	vmovaps %xmm10,(%r8)
  401f78:	c4 e2 79 18 3a       	vbroadcastss (%rdx),%xmm7
  401f7d:	c4 e2 79 18 37       	vbroadcastss (%rdi),%xmm6
  401f82:	c5 78 28 cf          	vmovaps %xmm7,%xmm9
  401f86:	c4 e2 79 18 18       	vbroadcastss (%rax),%xmm3
  401f8b:	c5 78 28 c6          	vmovaps %xmm6,%xmm8
  401f8f:	c5 f8 28 49 30       	vmovaps 0x30(%rcx),%xmm1
  401f94:	c5 f8 28 51 20       	vmovaps 0x20(%rcx),%xmm2
  401f99:	c4 e2 79 98 f9       	vfmadd132ps %xmm1,%xmm0,%xmm7
  401f9e:	c4 e2 79 98 f1       	vfmadd132ps %xmm1,%xmm0,%xmm6
  401fa3:	c5 78 28 df          	vmovaps %xmm7,%xmm11
  401fa7:	c4 e2 79 98 cb       	vfmadd132ps %xmm3,%xmm0,%xmm1
  401fac:	c5 78 28 d6          	vmovaps %xmm6,%xmm10
  401fb0:	c4 e2 79 18 7a 04    	vbroadcastss 0x4(%rdx),%xmm7
  401fb6:	c4 62 79 98 ca       	vfmadd132ps %xmm2,%xmm0,%xmm9
  401fbb:	c4 62 79 98 c2       	vfmadd132ps %xmm2,%xmm0,%xmm8
  401fc0:	c4 e2 79 98 d3       	vfmadd132ps %xmm3,%xmm0,%xmm2
  401fc5:	c5 78 28 e2          	vmovaps %xmm2,%xmm12
  401fc9:	c4 e2 79 18 58 04    	vbroadcastss 0x4(%rax),%xmm3
  401fcf:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  401fd4:	c5 f8 28 51 60       	vmovaps 0x60(%rcx),%xmm2
  401fd9:	c4 e2 79 18 77 04    	vbroadcastss 0x4(%rdi),%xmm6
  401fdf:	c4 62 41 b8 ca       	vfmadd231ps %xmm2,%xmm7,%xmm9
  401fe4:	c4 62 61 b8 e2       	vfmadd231ps %xmm2,%xmm3,%xmm12
  401fe9:	c5 f8 28 61 70       	vmovaps 0x70(%rcx),%xmm4
  401fee:	c4 62 49 b8 c2       	vfmadd231ps %xmm2,%xmm6,%xmm8
  401ff3:	c4 c2 79 18 6d 00    	vbroadcastss 0x0(%r13),%xmm5
  401ff9:	c4 62 41 b8 dc       	vfmadd231ps %xmm4,%xmm7,%xmm11
  401ffe:	c4 62 49 b8 d4       	vfmadd231ps %xmm4,%xmm6,%xmm10
  402003:	c4 e2 71 98 e3       	vfmadd132ps %xmm3,%xmm1,%xmm4
  402008:	c5 f8 28 91 a0 00 00 	vmovaps 0xa0(%rcx),%xmm2
  40200f:	00 
  402010:	c4 c2 79 18 3f       	vbroadcastss (%r15),%xmm7
  402015:	c4 62 51 b8 e2       	vfmadd231ps %xmm2,%xmm5,%xmm12
  40201a:	c4 c2 79 18 36       	vbroadcastss (%r14),%xmm6
  40201f:	c4 62 41 b8 ca       	vfmadd231ps %xmm2,%xmm7,%xmm9
  402024:	c5 f8 28 99 b0 00 00 	vmovaps 0xb0(%rcx),%xmm3
  40202b:	00 
  40202c:	c4 62 49 b8 c2       	vfmadd231ps %xmm2,%xmm6,%xmm8
  402031:	c4 62 41 b8 db       	vfmadd231ps %xmm3,%xmm7,%xmm11
  402036:	c4 62 49 b8 d3       	vfmadd231ps %xmm3,%xmm6,%xmm10
  40203b:	c4 e2 59 98 dd       	vfmadd132ps %xmm5,%xmm4,%xmm3
  402040:	c4 e2 79 18 28       	vbroadcastss (%rax),%xmm5
  402045:	8b 04 24             	mov    (%rsp),%eax
  402048:	c5 f8 28 91 e0 00 00 	vmovaps 0xe0(%rcx),%xmm2
  40204f:	00 
  402050:	c5 f8 28 89 f0 00 00 	vmovaps 0xf0(%rcx),%xmm1
  402057:	00 
  402058:	85 c0                	test   %eax,%eax
  40205a:	c4 c2 79 18 3c 24    	vbroadcastss (%r12),%xmm7
  402060:	c4 e2 79 18 33       	vbroadcastss (%rbx),%xmm6
  402065:	c4 62 41 b8 ca       	vfmadd231ps %xmm2,%xmm7,%xmm9
  40206a:	c4 e2 21 98 f9       	vfmadd132ps %xmm1,%xmm11,%xmm7
  40206f:	c4 41 78 29 0c 31    	vmovaps %xmm9,(%r9,%rsi,1)
  402075:	c4 62 49 b8 c2       	vfmadd231ps %xmm2,%xmm6,%xmm8
  40207a:	c4 e2 19 98 d5       	vfmadd132ps %xmm5,%xmm12,%xmm2
  40207f:	c4 e2 29 98 f1       	vfmadd132ps %xmm1,%xmm10,%xmm6
  402084:	c4 e2 61 98 e9       	vfmadd132ps %xmm1,%xmm3,%xmm5
  402089:	c4 41 78 29 04 32    	vmovaps %xmm8,(%r10,%rsi,1)
  40208f:	c4 c1 78 29 14 30    	vmovaps %xmm2,(%r8,%rsi,1)
  402095:	c4 c1 78 29 34 72    	vmovaps %xmm6,(%r10,%rsi,2)
  40209b:	c4 c1 78 29 3c 71    	vmovaps %xmm7,(%r9,%rsi,2)
  4020a1:	c4 c1 78 29 2c 70    	vmovaps %xmm5,(%r8,%rsi,2)
  4020a7:	75 2d                	jne    4020d6 <main._omp_fn.0+0x3d6>
  4020a9:	0f 31                	rdtsc  
  4020ab:	48 c1 e2 20          	shl    $0x20,%rdx
  4020af:	89 c0                	mov    %eax,%eax
  4020b1:	48 09 c2             	or     %rax,%rdx
  4020b4:	0f 88 a6 00 00 00    	js     402160 <main._omp_fn.0+0x460>
  4020ba:	c4 e1 f3 2a ca       	vcvtsi2sd %rdx,%xmm1,%xmm1
  4020bf:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
  4020c4:	c5 fb 11 48 10       	vmovsd %xmm1,0x10(%rax)
  4020c9:	c5 f3 5c 48 08       	vsubsd 0x8(%rax),%xmm1,%xmm1
  4020ce:	c5 f3 58 08          	vaddsd (%rax),%xmm1,%xmm1
  4020d2:	c5 fb 11 08          	vmovsd %xmm1,(%rax)
  4020d6:	83 44 24 04 01       	addl   $0x1,0x4(%rsp)
  4020db:	83 44 24 18 02       	addl   $0x2,0x18(%rsp)
  4020e0:	8b 44 24 24          	mov    0x24(%rsp),%eax
  4020e4:	39 44 24 04          	cmp    %eax,0x4(%rsp)
  4020e8:	74 5e                	je     402148 <main._omp_fn.0+0x448>
  4020ea:	8b 44 24 18          	mov    0x18(%rsp),%eax
  4020ee:	99                   	cltd   
  4020ef:	f7 7c 24 1c          	idivl  0x1c(%rsp)
  4020f3:	c1 e0 02             	shl    $0x2,%eax
  4020f6:	0f af 44 24 20       	imul   0x20(%rsp),%eax
  4020fb:	8d 78 03             	lea    0x3(%rax),%edi
  4020fe:	85 c0                	test   %eax,%eax
  402100:	44 8d 0c 50          	lea    (%rax,%rdx,2),%r9d
  402104:	0f 49 f8             	cmovns %eax,%edi
  402107:	c1 ff 02             	sar    $0x2,%edi
  40210a:	01 d7                	add    %edx,%edi
  40210c:	8b 14 24             	mov    (%rsp),%edx
  40210f:	85 d2                	test   %edx,%edx
  402111:	0f 85 e0 fc ff ff    	jne    401df7 <main._omp_fn.0+0xf7>
  402117:	0f 31                	rdtsc  
  402119:	48 c1 e2 20          	shl    $0x20,%rdx
  40211d:	89 c0                	mov    %eax,%eax
  40211f:	48 09 c2             	or     %rax,%rdx
  402122:	0f 89 c0 fc ff ff    	jns    401de8 <main._omp_fn.0+0xe8>
  402128:	49 89 d0             	mov    %rdx,%r8
  40212b:	83 e2 01             	and    $0x1,%edx
  40212e:	49 d1 e8             	shr    %r8
  402131:	49 09 d0             	or     %rdx,%r8
  402134:	c4 c1 f3 2a c8       	vcvtsi2sd %r8,%xmm1,%xmm1
  402139:	c5 f3 58 c9          	vaddsd %xmm1,%xmm1,%xmm1
  40213d:	e9 ab fc ff ff       	jmpq   401ded <main._omp_fn.0+0xed>
  402142:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  402148:	48 83 c4 68          	add    $0x68,%rsp
  40214c:	5b                   	pop    %rbx
  40214d:	5d                   	pop    %rbp
  40214e:	41 5c                	pop    %r12
  402150:	41 5d                	pop    %r13
  402152:	41 5e                	pop    %r14
  402154:	41 5f                	pop    %r15
  402156:	c3                   	retq   
  402157:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40215e:	00 00 
  402160:	48 89 d7             	mov    %rdx,%rdi
  402163:	83 e2 01             	and    $0x1,%edx
  402166:	48 d1 ef             	shr    %rdi
  402169:	48 09 d7             	or     %rdx,%rdi
  40216c:	c4 e1 f3 2a cf       	vcvtsi2sd %rdi,%xmm1,%xmm1
  402171:	c5 f3 58 c9          	vaddsd %xmm1,%xmm1,%xmm1
  402175:	e9 45 ff ff ff       	jmpq   4020bf <main._omp_fn.0+0x3bf>
  40217a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000402180 <_Z20generateCoefficientsPf>:
  402180:	55                   	push   %rbp
  402181:	48 b8 00 00 00 00 ff 	movabs $0xffffffff00000000,%rax
  402188:	ff ff ff 
  40218b:	48 be 00 00 00 00 00 	movabs $0x3f80000000000000,%rsi
  402192:	00 80 3f 
  402195:	49 b8 00 00 00 00 ab 	movabs $0x3f2aaaab00000000,%r8
  40219c:	aa 2a 3f 
  40219f:	48 b9 00 00 00 00 ab 	movabs $0x3eaaaaab00000000,%rcx
  4021a6:	aa aa 3e 
  4021a9:	49 b9 00 00 00 00 39 	movabs $0x3e638e3900000000,%r9
  4021b0:	8e 63 3e 
  4021b3:	48 89 e5             	mov    %rsp,%rbp
  4021b6:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4021ba:	49 ba 00 00 00 00 39 	movabs $0x3de38e3900000000,%r10
  4021c1:	8e e3 3d 
  4021c4:	48 83 ec 48          	sub    $0x48,%rsp
  4021c8:	49 bb 00 00 00 00 39 	movabs $0x3ee38e3900000000,%r11
  4021cf:	8e e3 3e 
  4021d2:	48 c7 44 24 c0 00 00 	movq   $0x0,-0x40(%rsp)
  4021d9:	00 00 
  4021db:	48 8b 54 24 c0       	mov    -0x40(%rsp),%rdx
  4021e0:	48 c7 44 24 b0 00 00 	movq   $0x0,-0x50(%rsp)
  4021e7:	00 00 
  4021e9:	48 c7 44 24 a0 00 00 	movq   $0x0,-0x60(%rsp)
  4021f0:	00 00 
  4021f2:	c5 f8 28 05 76 29 00 	vmovaps 0x2976(%rip),%xmm0        # 404b70 <__dso_handle+0x78>
  4021f9:	00 
  4021fa:	48 21 c2             	and    %rax,%rdx
  4021fd:	48 c7 44 24 b8 00 00 	movq   $0x0,-0x48(%rsp)
  402204:	00 00 
  402206:	48 c7 44 24 a8 00 00 	movq   $0x0,-0x58(%rsp)
  40220d:	00 00 
  40220f:	48 81 ca ab aa 2a 3f 	or     $0x3f2aaaab,%rdx
  402216:	48 89 4c 24 b8       	mov    %rcx,-0x48(%rsp)
  40221b:	4c 89 4c 24 a8       	mov    %r9,-0x58(%rsp)
  402220:	89 d2                	mov    %edx,%edx
  402222:	48 c7 44 24 98 00 00 	movq   $0x0,-0x68(%rsp)
  402229:	00 00 
  40222b:	48 c7 44 24 88 00 00 	movq   $0x0,-0x78(%rsp)
  402232:	00 00 
  402234:	48 09 f2             	or     %rsi,%rdx
  402237:	48 c7 44 24 90 00 00 	movq   $0x0,-0x70(%rsp)
  40223e:	00 00 
  402240:	4c 89 54 24 98       	mov    %r10,-0x68(%rsp)
  402245:	48 89 54 24 c0       	mov    %rdx,-0x40(%rsp)
  40224a:	48 8b 54 24 b0       	mov    -0x50(%rsp),%rdx
  40224f:	c5 f8 29 07          	vmovaps %xmm0,(%rdi)
  402253:	48 c7 44 24 88 00 00 	movq   $0x0,-0x78(%rsp)
  40225a:	00 00 
  40225c:	48 c7 44 24 90 00 00 	movq   $0x0,-0x70(%rsp)
  402263:	00 00 
  402265:	48 21 c2             	and    %rax,%rdx
  402268:	48 81 ca 39 8e e3 3e 	or     $0x3ee38e39,%rdx
  40226f:	89 d2                	mov    %edx,%edx
  402271:	c5 f8 28 05 07 29 00 	vmovaps 0x2907(%rip),%xmm0        # 404b80 <__dso_handle+0x88>
  402278:	00 
  402279:	4c 09 c2             	or     %r8,%rdx
  40227c:	48 89 54 24 b0       	mov    %rdx,-0x50(%rsp)
  402281:	48 8b 54 24 a0       	mov    -0x60(%rsp),%rdx
  402286:	c5 f8 29 47 10       	vmovaps %xmm0,0x10(%rdi)
  40228b:	48 21 c2             	and    %rax,%rdx
  40228e:	48 81 ca 39 8e 63 3e 	or     $0x3e638e39,%rdx
  402295:	c5 f8 28 05 f3 28 00 	vmovaps 0x28f3(%rip),%xmm0        # 404b90 <__dso_handle+0x98>
  40229c:	00 
  40229d:	89 d2                	mov    %edx,%edx
  40229f:	48 09 ca             	or     %rcx,%rdx
  4022a2:	c5 f8 29 47 20       	vmovaps %xmm0,0x20(%rdi)
  4022a7:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  4022ab:	48 89 54 24 a0       	mov    %rdx,-0x60(%rsp)
  4022b0:	c5 f8 28 4c 24 b8    	vmovaps -0x48(%rsp),%xmm1
  4022b6:	c5 f8 28 54 24 a8    	vmovaps -0x58(%rsp),%xmm2
  4022bc:	c5 f8 29 4f 40       	vmovaps %xmm1,0x40(%rdi)
  4022c1:	c5 f8 29 47 30       	vmovaps %xmm0,0x30(%rdi)
  4022c6:	c5 f8 29 57 50       	vmovaps %xmm2,0x50(%rdi)
  4022cb:	48 c7 44 24 a8 00 00 	movq   $0x0,-0x58(%rsp)
  4022d2:	00 00 
  4022d4:	48 8b 54 24 a8       	mov    -0x58(%rsp),%rdx
  4022d9:	48 c7 44 24 b0 00 00 	movq   $0x0,-0x50(%rsp)
  4022e0:	00 00 
  4022e2:	48 c7 44 24 b8 00 00 	movq   $0x0,-0x48(%rsp)
  4022e9:	00 00 
  4022eb:	c5 f8 28 5c 24 98    	vmovaps -0x68(%rsp),%xmm3
  4022f1:	48 21 c2             	and    %rax,%rdx
  4022f4:	48 c7 44 24 98 00 00 	movq   $0x0,-0x68(%rsp)
  4022fb:	00 00 
  4022fd:	48 c7 44 24 a0 00 00 	movq   $0x0,-0x60(%rsp)
  402304:	00 00 
  402306:	48 81 ca ab aa aa 3e 	or     $0x3eaaaaab,%rdx
  40230d:	c5 f8 28 64 24 88    	vmovaps -0x78(%rsp),%xmm4
  402313:	89 d2                	mov    %edx,%edx
  402315:	48 c7 44 24 88 00 00 	movq   $0x0,-0x78(%rsp)
  40231c:	00 00 
  40231e:	48 c7 44 24 90 00 00 	movq   $0x0,-0x70(%rsp)
  402325:	00 00 
  402327:	4c 09 ca             	or     %r9,%rdx
  40232a:	48 c7 44 24 c0 00 00 	movq   $0x0,-0x40(%rsp)
  402331:	00 00 
  402333:	48 c7 44 24 b8 00 00 	movq   $0x0,-0x48(%rsp)
  40233a:	00 00 
  40233c:	48 89 54 24 a8       	mov    %rdx,-0x58(%rsp)
  402341:	48 8b 54 24 b0       	mov    -0x50(%rsp),%rdx
  402346:	48 c7 44 24 c0 00 00 	movq   $0x0,-0x40(%rsp)
  40234d:	00 00 
  40234f:	c5 f8 28 6c 24 b8    	vmovaps -0x48(%rsp),%xmm5
  402355:	48 21 c2             	and    %rax,%rdx
  402358:	48 81 ca 39 8e e3 3d 	or     $0x3de38e39,%rdx
  40235f:	81 e2 39 8e e3 3d    	and    $0x3de38e39,%edx
  402365:	48 89 54 24 b0       	mov    %rdx,-0x50(%rsp)
  40236a:	48 8b 54 24 98       	mov    -0x68(%rsp),%rdx
  40236f:	c5 f8 28 74 24 a8    	vmovaps -0x58(%rsp),%xmm6
  402375:	48 21 c2             	and    %rax,%rdx
  402378:	48 81 ca ab aa 2a 3f 	or     $0x3f2aaaab,%rdx
  40237f:	89 d2                	mov    %edx,%edx
  402381:	4c 09 da             	or     %r11,%rdx
  402384:	48 89 54 24 98       	mov    %rdx,-0x68(%rsp)
  402389:	48 8b 54 24 a0       	mov    -0x60(%rsp),%rdx
  40238e:	c5 f8 29 5f 60       	vmovaps %xmm3,0x60(%rdi)
  402393:	48 21 c2             	and    %rax,%rdx
  402396:	48 81 ca 39 8e 63 3e 	or     $0x3e638e39,%rdx
  40239d:	81 e2 39 8e 63 3e    	and    $0x3e638e39,%edx
  4023a3:	48 89 54 24 a0       	mov    %rdx,-0x60(%rsp)
  4023a8:	48 8b 54 24 88       	mov    -0x78(%rsp),%rdx
  4023ad:	c5 f8 28 7c 24 98    	vmovaps -0x68(%rsp),%xmm7
  4023b3:	48 21 c2             	and    %rax,%rdx
  4023b6:	48 81 ca 00 00 80 3f 	or     $0x3f800000,%rdx
  4023bd:	89 d2                	mov    %edx,%edx
  4023bf:	4c 09 c2             	or     %r8,%rdx
  4023c2:	48 89 54 24 88       	mov    %rdx,-0x78(%rsp)
  4023c7:	48 8b 54 24 90       	mov    -0x70(%rsp),%rdx
  4023cc:	c5 f8 29 67 70       	vmovaps %xmm4,0x70(%rdi)
  4023d1:	48 21 c2             	and    %rax,%rdx
  4023d4:	48 81 ca ab aa aa 3e 	or     $0x3eaaaaab,%rdx
  4023db:	81 e2 ab aa aa 3e    	and    $0x3eaaaaab,%edx
  4023e1:	48 89 54 24 90       	mov    %rdx,-0x70(%rsp)
  4023e6:	c5 f8 28 4c 24 88    	vmovaps -0x78(%rsp),%xmm1
  4023ec:	c5 f8 29 af 80 00 00 	vmovaps %xmm5,0x80(%rdi)
  4023f3:	00 
  4023f4:	c5 f8 29 b7 90 00 00 	vmovaps %xmm6,0x90(%rdi)
  4023fb:	00 
  4023fc:	c5 f8 29 bf a0 00 00 	vmovaps %xmm7,0xa0(%rdi)
  402403:	00 
  402404:	c5 f8 29 8f b0 00 00 	vmovaps %xmm1,0xb0(%rdi)
  40240b:	00 
  40240c:	48 c7 44 24 b0 00 00 	movq   $0x0,-0x50(%rsp)
  402413:	00 00 
  402415:	48 8b 54 24 b0       	mov    -0x50(%rsp),%rdx
  40241a:	48 c7 44 24 a0 00 00 	movq   $0x0,-0x60(%rsp)
  402421:	00 00 
  402423:	48 c7 44 24 90 00 00 	movq   $0x0,-0x70(%rsp)
  40242a:	00 00 
  40242c:	48 c7 44 24 b8 00 00 	movq   $0x0,-0x48(%rsp)
  402433:	00 00 
  402435:	48 c7 44 24 c0 00 00 	movq   $0x0,-0x40(%rsp)
  40243c:	00 00 
  40243e:	48 21 c2             	and    %rax,%rdx
  402441:	48 c7 44 24 a8 00 00 	movq   $0x0,-0x58(%rsp)
  402448:	00 00 
  40244a:	48 c7 44 24 98 00 00 	movq   $0x0,-0x68(%rsp)
  402451:	00 00 
  402453:	48 81 ca 39 8e 63 3e 	or     $0x3e638e39,%rdx
  40245a:	48 c7 44 24 88 00 00 	movq   $0x0,-0x78(%rsp)
  402461:	00 00 
  402463:	48 c7 44 24 b8 00 00 	movq   $0x0,-0x48(%rsp)
  40246a:	00 00 
  40246c:	89 d2                	mov    %edx,%edx
  40246e:	48 c7 44 24 c0 00 00 	movq   $0x0,-0x40(%rsp)
  402475:	00 00 
  402477:	4c 89 54 24 a8       	mov    %r10,-0x58(%rsp)
  40247c:	48 09 ca             	or     %rcx,%rdx
  40247f:	4c 89 4c 24 98       	mov    %r9,-0x68(%rsp)
  402484:	48 89 4c 24 88       	mov    %rcx,-0x78(%rsp)
  402489:	48 89 54 24 b0       	mov    %rdx,-0x50(%rsp)
  40248e:	48 8b 54 24 a0       	mov    -0x60(%rsp),%rdx
  402493:	c5 f8 28 54 24 b8    	vmovaps -0x48(%rsp),%xmm2
  402499:	48 21 c2             	and    %rax,%rdx
  40249c:	c5 f8 28 5c 24 a8    	vmovaps -0x58(%rsp),%xmm3
  4024a2:	48 81 ca 39 8e e3 3e 	or     $0x3ee38e39,%rdx
  4024a9:	89 d2                	mov    %edx,%edx
  4024ab:	4c 09 c2             	or     %r8,%rdx
  4024ae:	48 89 54 24 a0       	mov    %rdx,-0x60(%rsp)
  4024b3:	48 8b 54 24 90       	mov    -0x70(%rsp),%rdx
  4024b8:	c5 f8 28 64 24 98    	vmovaps -0x68(%rsp),%xmm4
  4024be:	48 21 c2             	and    %rax,%rdx
  4024c1:	48 81 ca ab aa 2a 3f 	or     $0x3f2aaaab,%rdx
  4024c8:	89 d2                	mov    %edx,%edx
  4024ca:	48 09 f2             	or     %rsi,%rdx
  4024cd:	48 89 54 24 90       	mov    %rdx,-0x70(%rsp)
  4024d2:	c5 f8 28 6c 24 88    	vmovaps -0x78(%rsp),%xmm5
  4024d8:	c5 f8 29 97 c0 00 00 	vmovaps %xmm2,0xc0(%rdi)
  4024df:	00 
  4024e0:	c5 f8 29 9f d0 00 00 	vmovaps %xmm3,0xd0(%rdi)
  4024e7:	00 
  4024e8:	c5 f8 29 a7 e0 00 00 	vmovaps %xmm4,0xe0(%rdi)
  4024ef:	00 
  4024f0:	c5 f8 29 af f0 00 00 	vmovaps %xmm5,0xf0(%rdi)
  4024f7:	00 
  4024f8:	c9                   	leaveq 
  4024f9:	c3                   	retq   
  4024fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000402500 <_Z6kernelPfS_S_S_S_S_S_i>:
  402500:	41 57                	push   %r15
  402502:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  402506:	c4 62 79 18 0f       	vbroadcastss (%rdi),%xmm9
  40250b:	41 56                	push   %r14
  40250d:	c4 62 79 18 06       	vbroadcastss (%rsi),%xmm8
  402512:	41 55                	push   %r13
  402514:	c4 e2 79 18 12       	vbroadcastss (%rdx),%xmm2
  402519:	41 54                	push   %r12
  40251b:	55                   	push   %rbp
  40251c:	53                   	push   %rbx
  40251d:	4c 63 5c 24 40       	movslq 0x40(%rsp),%r11
  402522:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
  402527:	c4 e2 79 18 77 04    	vbroadcastss 0x4(%rdi),%xmm6
  40252d:	44 89 db             	mov    %r11d,%ebx
  402530:	c5 f8 28 08          	vmovaps (%rax),%xmm1
  402534:	c1 eb 1f             	shr    $0x1f,%ebx
  402537:	c5 78 28 e6          	vmovaps %xmm6,%xmm12
  40253b:	c4 e2 79 18 6e 04    	vbroadcastss 0x4(%rsi),%xmm5
  402541:	44 01 db             	add    %r11d,%ebx
  402544:	c4 e2 79 98 d1       	vfmadd132ps %xmm1,%xmm0,%xmm2
  402549:	d1 fb                	sar    %ebx
  40254b:	c4 62 79 98 c9       	vfmadd132ps %xmm1,%xmm0,%xmm9
  402550:	c4 62 79 98 c1       	vfmadd132ps %xmm1,%xmm0,%xmm8
  402555:	c5 f8 28 60 10       	vmovaps 0x10(%rax),%xmm4
  40255a:	c5 78 28 dd          	vmovaps %xmm5,%xmm11
  40255e:	48 63 db             	movslq %ebx,%rbx
  402561:	4e 8d 14 9d 00 00 00 	lea    0x0(,%r11,4),%r10
  402568:	00 
  402569:	c4 e2 79 18 4a 04    	vbroadcastss 0x4(%rdx),%xmm1
  40256f:	48 c1 e3 02          	shl    $0x2,%rbx
  402573:	c4 62 79 98 e4       	vfmadd132ps %xmm4,%xmm0,%xmm12
  402578:	c4 62 79 98 dc       	vfmadd132ps %xmm4,%xmm0,%xmm11
  40257d:	4c 8d 34 1f          	lea    (%rdi,%rbx,1),%r14
  402581:	4c 8d 2c 1e          	lea    (%rsi,%rbx,1),%r13
  402585:	c5 f8 28 78 40       	vmovaps 0x40(%rax),%xmm7
  40258a:	c4 e2 79 98 e1       	vfmadd132ps %xmm1,%xmm0,%xmm4
  40258f:	4c 8d 24 1a          	lea    (%rdx,%rbx,1),%r12
  402593:	c5 78 28 d4          	vmovaps %xmm4,%xmm10
  402597:	48 83 c3 04          	add    $0x4,%rbx
  40259b:	c5 f8 28 60 50       	vmovaps 0x50(%rax),%xmm4
  4025a0:	4c 8d 3c 1f          	lea    (%rdi,%rbx,1),%r15
  4025a4:	48 8d 2c 1e          	lea    (%rsi,%rbx,1),%rbp
  4025a8:	c4 62 49 b8 cf       	vfmadd231ps %xmm7,%xmm6,%xmm9
  4025ad:	c4 62 51 b8 c7       	vfmadd231ps %xmm7,%xmm5,%xmm8
  4025b2:	c4 e2 69 98 f9       	vfmadd132ps %xmm1,%xmm2,%xmm7
  4025b7:	c5 f8 28 98 90 00 00 	vmovaps 0x90(%rax),%xmm3
  4025be:	00 
  4025bf:	c4 e2 19 98 f4       	vfmadd132ps %xmm4,%xmm12,%xmm6
  4025c4:	c4 e2 21 98 ec       	vfmadd132ps %xmm4,%xmm11,%xmm5
  4025c9:	c4 e2 29 98 e1       	vfmadd132ps %xmm1,%xmm10,%xmm4
  4025ce:	48 01 d3             	add    %rdx,%rbx
  4025d1:	c4 42 79 18 26       	vbroadcastss (%r14),%xmm12
  4025d6:	c4 42 79 18 5d 00    	vbroadcastss 0x0(%r13),%xmm11
  4025dc:	c4 e2 19 b8 f3       	vfmadd231ps %xmm3,%xmm12,%xmm6
  4025e1:	c4 42 79 18 14 24    	vbroadcastss (%r12),%xmm10
  4025e7:	c4 e2 21 b8 eb       	vfmadd231ps %xmm3,%xmm11,%xmm5
  4025ec:	c5 f8 28 90 80 00 00 	vmovaps 0x80(%rax),%xmm2
  4025f3:	00 
  4025f4:	c4 e2 29 b8 e3       	vfmadd231ps %xmm3,%xmm10,%xmm4
  4025f9:	c4 e2 29 b8 fa       	vfmadd231ps %xmm2,%xmm10,%xmm7
  4025fe:	c4 62 19 b8 ca       	vfmadd231ps %xmm2,%xmm12,%xmm9
  402603:	c4 62 79 18 13       	vbroadcastss (%rbx),%xmm10
  402608:	c4 62 21 b8 c2       	vfmadd231ps %xmm2,%xmm11,%xmm8
  40260d:	c4 42 79 18 27       	vbroadcastss (%r15),%xmm12
  402612:	c4 62 79 18 5d 00    	vbroadcastss 0x0(%rbp),%xmm11
  402618:	c5 f8 28 90 c0 00 00 	vmovaps 0xc0(%rax),%xmm2
  40261f:	00 
  402620:	c5 f8 28 88 d0 00 00 	vmovaps 0xd0(%rax),%xmm1
  402627:	00 
  402628:	c4 62 19 b8 ca       	vfmadd231ps %xmm2,%xmm12,%xmm9
  40262d:	c4 62 21 b8 c2       	vfmadd231ps %xmm2,%xmm11,%xmm8
  402632:	c4 c2 41 98 d2       	vfmadd132ps %xmm10,%xmm7,%xmm2
  402637:	c5 78 29 09          	vmovaps %xmm9,(%rcx)
  40263b:	c4 62 49 98 e1       	vfmadd132ps %xmm1,%xmm6,%xmm12
  402640:	c4 62 51 98 d9       	vfmadd132ps %xmm1,%xmm5,%xmm11
  402645:	4c 01 d1             	add    %r10,%rcx
  402648:	c4 62 59 98 d1       	vfmadd132ps %xmm1,%xmm4,%xmm10
  40264d:	c4 41 78 29 00       	vmovaps %xmm8,(%r8)
  402652:	4d 01 d0             	add    %r10,%r8
  402655:	c4 c1 78 29 11       	vmovaps %xmm2,(%r9)
  40265a:	4d 01 d1             	add    %r10,%r9
  40265d:	c5 78 29 21          	vmovaps %xmm12,(%rcx)
  402661:	4c 01 d1             	add    %r10,%rcx
  402664:	c4 41 78 29 18       	vmovaps %xmm11,(%r8)
  402669:	4d 01 d0             	add    %r10,%r8
  40266c:	4d 01 ca             	add    %r9,%r10
  40266f:	c4 41 78 29 11       	vmovaps %xmm10,(%r9)
  402674:	c4 e2 79 18 3f       	vbroadcastss (%rdi),%xmm7
  402679:	c4 e2 79 18 36       	vbroadcastss (%rsi),%xmm6
  40267e:	c5 78 28 cf          	vmovaps %xmm7,%xmm9
  402682:	c5 f8 28 48 30       	vmovaps 0x30(%rax),%xmm1
  402687:	c5 78 28 c6          	vmovaps %xmm6,%xmm8
  40268b:	c4 e2 79 18 1a       	vbroadcastss (%rdx),%xmm3
  402690:	c4 e2 79 98 f9       	vfmadd132ps %xmm1,%xmm0,%xmm7
  402695:	c4 e2 79 98 f1       	vfmadd132ps %xmm1,%xmm0,%xmm6
  40269a:	c5 78 28 df          	vmovaps %xmm7,%xmm11
  40269e:	c5 78 28 d6          	vmovaps %xmm6,%xmm10
  4026a2:	c5 f8 28 50 20       	vmovaps 0x20(%rax),%xmm2
  4026a7:	c4 e2 79 18 7f 04    	vbroadcastss 0x4(%rdi),%xmm7
  4026ad:	c4 62 79 98 ca       	vfmadd132ps %xmm2,%xmm0,%xmm9
  4026b2:	c4 62 79 98 c2       	vfmadd132ps %xmm2,%xmm0,%xmm8
  4026b7:	c4 e2 79 98 d3       	vfmadd132ps %xmm3,%xmm0,%xmm2
  4026bc:	c4 e2 61 b8 c1       	vfmadd231ps %xmm1,%xmm3,%xmm0
  4026c1:	c5 78 28 e2          	vmovaps %xmm2,%xmm12
  4026c5:	c4 e2 79 18 76 04    	vbroadcastss 0x4(%rsi),%xmm6
  4026cb:	c5 f8 28 50 60       	vmovaps 0x60(%rax),%xmm2
  4026d0:	c4 e2 79 18 5a 04    	vbroadcastss 0x4(%rdx),%xmm3
  4026d6:	c4 62 41 b8 ca       	vfmadd231ps %xmm2,%xmm7,%xmm9
  4026db:	c4 62 49 b8 c2       	vfmadd231ps %xmm2,%xmm6,%xmm8
  4026e0:	c5 f8 28 60 70       	vmovaps 0x70(%rax),%xmm4
  4026e5:	c4 62 61 b8 e2       	vfmadd231ps %xmm2,%xmm3,%xmm12
  4026ea:	c4 c2 79 18 2c 24    	vbroadcastss (%r12),%xmm5
  4026f0:	c4 62 41 b8 dc       	vfmadd231ps %xmm4,%xmm7,%xmm11
  4026f5:	c4 62 49 b8 d4       	vfmadd231ps %xmm4,%xmm6,%xmm10
  4026fa:	c4 e2 79 98 e3       	vfmadd132ps %xmm3,%xmm0,%xmm4
  4026ff:	c4 c2 79 18 3e       	vbroadcastss (%r14),%xmm7
  402704:	c4 c2 79 18 75 00    	vbroadcastss 0x0(%r13),%xmm6
  40270a:	c5 f8 28 90 a0 00 00 	vmovaps 0xa0(%rax),%xmm2
  402711:	00 
  402712:	c5 f8 28 98 b0 00 00 	vmovaps 0xb0(%rax),%xmm3
  402719:	00 
  40271a:	c4 62 41 b8 ca       	vfmadd231ps %xmm2,%xmm7,%xmm9
  40271f:	c4 62 49 b8 c2       	vfmadd231ps %xmm2,%xmm6,%xmm8
  402724:	c4 62 51 b8 e2       	vfmadd231ps %xmm2,%xmm5,%xmm12
  402729:	c5 f8 28 88 f0 00 00 	vmovaps 0xf0(%rax),%xmm1
  402730:	00 
  402731:	c4 62 41 b8 db       	vfmadd231ps %xmm3,%xmm7,%xmm11
  402736:	c4 62 49 b8 d3       	vfmadd231ps %xmm3,%xmm6,%xmm10
  40273b:	c4 e2 59 98 dd       	vfmadd132ps %xmm5,%xmm4,%xmm3
  402740:	c4 c2 79 18 3f       	vbroadcastss (%r15),%xmm7
  402745:	c4 e2 79 18 75 00    	vbroadcastss 0x0(%rbp),%xmm6
  40274b:	c4 e2 79 18 2b       	vbroadcastss (%rbx),%xmm5
  402750:	c5 f8 28 90 e0 00 00 	vmovaps 0xe0(%rax),%xmm2
  402757:	00 
  402758:	c4 62 41 b8 ca       	vfmadd231ps %xmm2,%xmm7,%xmm9
  40275d:	c4 62 49 b8 c2       	vfmadd231ps %xmm2,%xmm6,%xmm8
  402762:	c5 78 29 09          	vmovaps %xmm9,(%rcx)
  402766:	c4 e2 19 98 d5       	vfmadd132ps %xmm5,%xmm12,%xmm2
  40276b:	c4 e2 21 98 f9       	vfmadd132ps %xmm1,%xmm11,%xmm7
  402770:	c4 e2 29 98 f1       	vfmadd132ps %xmm1,%xmm10,%xmm6
  402775:	c4 41 78 29 00       	vmovaps %xmm8,(%r8)
  40277a:	c4 e2 61 98 e9       	vfmadd132ps %xmm1,%xmm3,%xmm5
  40277f:	c4 c1 78 29 12       	vmovaps %xmm2,(%r10)
  402784:	c4 81 78 29 34 98    	vmovaps %xmm6,(%r8,%r11,4)
  40278a:	c4 a1 78 29 3c 99    	vmovaps %xmm7,(%rcx,%r11,4)
  402790:	c4 81 78 29 2c 9a    	vmovaps %xmm5,(%r10,%r11,4)
  402796:	5b                   	pop    %rbx
  402797:	5d                   	pop    %rbp
  402798:	41 5c                	pop    %r12
  40279a:	41 5d                	pop    %r13
  40279c:	41 5e                	pop    %r14
  40279e:	41 5f                	pop    %r15
  4027a0:	c3                   	retq   
  4027a1:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4027a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4027ad:	00 00 00 

00000000004027b0 <_Z11encodeImagePfS_S_>:
  4027b0:	55                   	push   %rbp
  4027b1:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  4027b5:	48 89 e5             	mov    %rsp,%rbp
  4027b8:	41 54                	push   %r12
  4027ba:	53                   	push   %rbx
  4027bb:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4027bf:	48 81 ec 10 02 00 00 	sub    $0x210,%rsp
  4027c6:	48 8d 84 24 98 00 00 	lea    0x98(%rsp),%rax
  4027cd:	00 
  4027ce:	c5 f9 6f 0d da 23 00 	vmovdqa 0x23da(%rip),%xmm1        # 404bb0 <__dso_handle+0xb8>
  4027d5:	00 
  4027d6:	48 89 bc 24 00 01 00 	mov    %rdi,0x100(%rsp)
  4027dd:	00 
  4027de:	48 89 bc 24 10 01 00 	mov    %rdi,0x110(%rsp)
  4027e5:	00 
  4027e6:	48 81 c7 00 40 00 00 	add    $0x4000,%rdi
  4027ed:	48 89 84 24 d0 00 00 	mov    %rax,0xd0(%rsp)
  4027f4:	00 
  4027f5:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
  4027fc:	00 
  4027fd:	48 89 bc 24 20 01 00 	mov    %rdi,0x120(%rsp)
  402804:	00 
  402805:	c5 f9 7f 84 24 e0 00 	vmovdqa %xmm0,0xe0(%rsp)
  40280c:	00 00 
  40280e:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
  402815:	00 
  402816:	48 8d 84 24 f8 00 00 	lea    0xf8(%rsp),%rax
  40281d:	00 
  40281e:	48 89 bc 24 18 01 00 	mov    %rdi,0x118(%rsp)
  402825:	00 
  402826:	c5 f9 7f 8c 24 f0 00 	vmovdqa %xmm1,0xf0(%rsp)
  40282d:	00 00 
  40282f:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  402834:	c5 f9 6f 05 64 23 00 	vmovdqa 0x2364(%rip),%xmm0        # 404ba0 <__dso_handle+0xa8>
  40283b:	00 
  40283c:	48 89 84 24 30 01 00 	mov    %rax,0x130(%rsp)
  402843:	00 
  402844:	48 8d 84 24 40 01 00 	lea    0x140(%rsp),%rax
  40284b:	00 
  40284c:	48 89 b4 24 60 01 00 	mov    %rsi,0x160(%rsp)
  402853:	00 
  402854:	c5 f9 7f 84 24 90 00 	vmovdqa %xmm0,0x90(%rsp)
  40285b:	00 00 
  40285d:	48 89 84 24 38 01 00 	mov    %rax,0x138(%rsp)
  402864:	00 
  402865:	48 8d 84 24 58 01 00 	lea    0x158(%rsp),%rax
  40286c:	00 
  40286d:	48 89 b4 24 70 01 00 	mov    %rsi,0x170(%rsp)
  402874:	00 
  402875:	48 81 c6 00 40 00 00 	add    $0x4000,%rsi
  40287c:	c5 f9 6f 05 3c 23 00 	vmovdqa 0x233c(%rip),%xmm0        # 404bc0 <__dso_handle+0xc8>
  402883:	00 
  402884:	48 c7 44 24 30 00 00 	movq   $0x0,0x30(%rsp)
  40288b:	00 00 
  40288d:	48 c7 44 24 38 00 00 	movq   $0x0,0x38(%rsp)
  402894:	00 00 
  402896:	48 c7 44 24 40 00 00 	movq   $0x0,0x40(%rsp)
  40289d:	00 00 
  40289f:	48 c7 84 24 c0 00 00 	movq   $0x0,0xc0(%rsp)
  4028a6:	00 00 00 00 00 
  4028ab:	48 c7 84 24 b8 00 00 	movq   $0x0,0xb8(%rsp)
  4028b2:	00 00 00 00 00 
  4028b7:	48 c7 84 24 b0 00 00 	movq   $0x0,0xb0(%rsp)
  4028be:	00 00 00 00 00 
  4028c3:	48 c7 84 24 a0 00 00 	movq   $0x0,0xa0(%rsp)
  4028ca:	00 00 00 00 00 
  4028cf:	48 c7 84 24 a8 00 00 	movq   $0x0,0xa8(%rsp)
  4028d6:	00 00 00 00 00 
  4028db:	48 c7 84 24 c8 00 00 	movq   $0x0,0xc8(%rsp)
  4028e2:	00 00 00 00 00 
  4028e7:	48 c7 84 24 08 01 00 	movq   $0x0,0x108(%rsp)
  4028ee:	00 00 00 00 00 
  4028f3:	48 c7 84 24 28 01 00 	movq   $0x0,0x128(%rsp)
  4028fa:	00 00 00 00 00 
  4028ff:	48 c7 84 24 68 01 00 	movq   $0x0,0x168(%rsp)
  402906:	00 00 00 00 00 
  40290b:	c5 f9 7f 84 24 40 01 	vmovdqa %xmm0,0x140(%rsp)
  402912:	00 00 
  402914:	48 c7 84 24 88 01 00 	movq   $0x0,0x188(%rsp)
  40291b:	00 00 00 00 00 
  402920:	48 89 84 24 90 01 00 	mov    %rax,0x190(%rsp)
  402927:	00 
  402928:	48 8d 84 24 a0 01 00 	lea    0x1a0(%rsp),%rax
  40292f:	00 
  402930:	c5 f9 7f 8c 24 50 01 	vmovdqa %xmm1,0x150(%rsp)
  402937:	00 00 
  402939:	c5 f9 7f 84 24 a0 01 	vmovdqa %xmm0,0x1a0(%rsp)
  402940:	00 00 
  402942:	c5 f9 7f 8c 24 b0 01 	vmovdqa %xmm1,0x1b0(%rsp)
  402949:	00 00 
  40294b:	c5 f9 7f 84 24 00 02 	vmovdqa %xmm0,0x200(%rsp)
  402952:	00 00 
  402954:	48 89 84 24 98 01 00 	mov    %rax,0x198(%rsp)
  40295b:	00 
  40295c:	48 8d 84 24 b8 01 00 	lea    0x1b8(%rsp),%rax
  402963:	00 
  402964:	48 89 94 24 c0 01 00 	mov    %rdx,0x1c0(%rsp)
  40296b:	00 
  40296c:	48 89 94 24 d0 01 00 	mov    %rdx,0x1d0(%rsp)
  402973:	00 
  402974:	48 81 c2 00 40 00 00 	add    $0x4000,%rdx
  40297b:	48 89 b4 24 80 01 00 	mov    %rsi,0x180(%rsp)
  402982:	00 
  402983:	48 89 84 24 f0 01 00 	mov    %rax,0x1f0(%rsp)
  40298a:	00 
  40298b:	48 89 94 24 e0 01 00 	mov    %rdx,0x1e0(%rsp)
  402992:	00 
  402993:	48 8d 84 24 00 02 00 	lea    0x200(%rsp),%rax
  40299a:	00 
  40299b:	48 89 94 24 d8 01 00 	mov    %rdx,0x1d8(%rsp)
  4029a2:	00 
  4029a3:	48 8d 94 24 b0 01 00 	lea    0x1b0(%rsp),%rdx
  4029aa:	00 
  4029ab:	48 89 b4 24 78 01 00 	mov    %rsi,0x178(%rsp)
  4029b2:	00 
  4029b3:	31 f6                	xor    %esi,%esi
  4029b5:	48 c7 84 24 c8 01 00 	movq   $0x0,0x1c8(%rsp)
  4029bc:	00 00 00 00 00 
  4029c1:	48 c7 84 24 e8 01 00 	movq   $0x0,0x1e8(%rsp)
  4029c8:	00 00 00 00 00 
  4029cd:	48 89 84 24 f8 01 00 	mov    %rax,0x1f8(%rsp)
  4029d4:	00 
  4029d5:	e8 d6 10 00 00       	callq  403ab0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_>
  4029da:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  4029df:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
  4029e4:	48 39 c7             	cmp    %rax,%rdi
  4029e7:	0f 84 72 05 00 00    	je     402f5f <_Z11encodeImagePfS_S_+0x7af>
  4029ed:	48 85 ff             	test   %rdi,%rdi
  4029f0:	0f 84 00 05 00 00    	je     402ef6 <_Z11encodeImagePfS_S_+0x746>
  4029f6:	c5 f9 6f 84 24 50 01 	vmovdqa 0x150(%rsp),%xmm0
  4029fd:	00 00 
  4029ff:	c5 fa 7f 07          	vmovdqu %xmm0,(%rdi)
  402a03:	48 8b 94 24 70 01 00 	mov    0x170(%rsp),%rdx
  402a0a:	00 
  402a0b:	48 8b 84 24 60 01 00 	mov    0x160(%rsp),%rax
  402a12:	00 
  402a13:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  402a17:	48 89 57 20          	mov    %rdx,0x20(%rdi)
  402a1b:	48 8b 94 24 78 01 00 	mov    0x178(%rsp),%rdx
  402a22:	00 
  402a23:	48 89 47 10          	mov    %rax,0x10(%rdi)
  402a27:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  402a2e:	00 
  402a2f:	c5 fa 7f 47 50       	vmovdqu %xmm0,0x50(%rdi)
  402a34:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  402a38:	48 8b 94 24 80 01 00 	mov    0x180(%rsp),%rdx
  402a3f:	00 
  402a40:	48 85 c0             	test   %rax,%rax
  402a43:	48 89 57 30          	mov    %rdx,0x30(%rdi)
  402a47:	48 8b 94 24 88 01 00 	mov    0x188(%rsp),%rdx
  402a4e:	00 
  402a4f:	48 89 47 18          	mov    %rax,0x18(%rdi)
  402a53:	48 89 57 38          	mov    %rdx,0x38(%rdi)
  402a57:	48 8d 57 08          	lea    0x8(%rdi),%rdx
  402a5b:	48 89 57 40          	mov    %rdx,0x40(%rdi)
  402a5f:	48 8d 57 50          	lea    0x50(%rdi),%rdx
  402a63:	48 89 57 48          	mov    %rdx,0x48(%rdi)
  402a67:	74 04                	je     402a6d <_Z11encodeImagePfS_S_+0x2bd>
  402a69:	f0 83 00 01          	lock addl $0x1,(%rax)
  402a6d:	83 bc 24 54 01 00 00 	cmpl   $0x2,0x154(%rsp)
  402a74:	02 
  402a75:	0f 8f 15 04 00 00    	jg     402e90 <_Z11encodeImagePfS_S_+0x6e0>
  402a7b:	48 8b 94 24 98 01 00 	mov    0x198(%rsp),%rdx
  402a82:	00 
  402a83:	48 8b 47 48          	mov    0x48(%rdi),%rax
  402a87:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  402a8c:	48 8b 12             	mov    (%rdx),%rdx
  402a8f:	48 89 10             	mov    %rdx,(%rax)
  402a92:	48 8b 94 24 98 01 00 	mov    0x198(%rsp),%rdx
  402a99:	00 
  402a9a:	48 8b 47 48          	mov    0x48(%rdi),%rax
  402a9e:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  402aa3:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  402aa7:	48 89 50 08          	mov    %rdx,0x8(%rax)
  402aab:	48 83 c7 60          	add    $0x60,%rdi
  402aaf:	48 89 7c 24 38       	mov    %rdi,0x38(%rsp)
  402ab4:	48 39 fe             	cmp    %rdi,%rsi
  402ab7:	0f 84 8b 04 00 00    	je     402f48 <_Z11encodeImagePfS_S_+0x798>
  402abd:	48 85 ff             	test   %rdi,%rdi
  402ac0:	0f 84 b0 00 00 00    	je     402b76 <_Z11encodeImagePfS_S_+0x3c6>
  402ac6:	c5 f9 6f 84 24 f0 00 	vmovdqa 0xf0(%rsp),%xmm0
  402acd:	00 00 
  402acf:	c5 fa 7f 07          	vmovdqu %xmm0,(%rdi)
  402ad3:	48 8b 94 24 10 01 00 	mov    0x110(%rsp),%rdx
  402ada:	00 
  402adb:	48 8b 84 24 00 01 00 	mov    0x100(%rsp),%rax
  402ae2:	00 
  402ae3:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  402ae7:	48 89 57 20          	mov    %rdx,0x20(%rdi)
  402aeb:	48 8b 94 24 18 01 00 	mov    0x118(%rsp),%rdx
  402af2:	00 
  402af3:	48 89 47 10          	mov    %rax,0x10(%rdi)
  402af7:	48 8b 84 24 08 01 00 	mov    0x108(%rsp),%rax
  402afe:	00 
  402aff:	c5 fa 7f 47 50       	vmovdqu %xmm0,0x50(%rdi)
  402b04:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  402b08:	48 8b 94 24 20 01 00 	mov    0x120(%rsp),%rdx
  402b0f:	00 
  402b10:	48 85 c0             	test   %rax,%rax
  402b13:	48 89 57 30          	mov    %rdx,0x30(%rdi)
  402b17:	48 8b 94 24 28 01 00 	mov    0x128(%rsp),%rdx
  402b1e:	00 
  402b1f:	48 89 47 18          	mov    %rax,0x18(%rdi)
  402b23:	48 89 57 38          	mov    %rdx,0x38(%rdi)
  402b27:	48 8d 57 08          	lea    0x8(%rdi),%rdx
  402b2b:	48 89 57 40          	mov    %rdx,0x40(%rdi)
  402b2f:	48 8d 57 50          	lea    0x50(%rdi),%rdx
  402b33:	48 89 57 48          	mov    %rdx,0x48(%rdi)
  402b37:	74 04                	je     402b3d <_Z11encodeImagePfS_S_+0x38d>
  402b39:	f0 83 00 01          	lock addl $0x1,(%rax)
  402b3d:	83 bc 24 f4 00 00 00 	cmpl   $0x2,0xf4(%rsp)
  402b44:	02 
  402b45:	0f 8f 8d 03 00 00    	jg     402ed8 <_Z11encodeImagePfS_S_+0x728>
  402b4b:	48 8b 94 24 38 01 00 	mov    0x138(%rsp),%rdx
  402b52:	00 
  402b53:	48 8b 47 48          	mov    0x48(%rdi),%rax
  402b57:	48 8b 12             	mov    (%rdx),%rdx
  402b5a:	48 89 10             	mov    %rdx,(%rax)
  402b5d:	48 8b 94 24 38 01 00 	mov    0x138(%rsp),%rdx
  402b64:	00 
  402b65:	48 8b 47 48          	mov    0x48(%rdi),%rax
  402b69:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  402b6e:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  402b72:	48 89 50 08          	mov    %rdx,0x8(%rax)
  402b76:	48 83 c7 60          	add    $0x60,%rdi
  402b7a:	48 89 7c 24 38       	mov    %rdi,0x38(%rsp)
  402b7f:	48 8d b4 24 90 00 00 	lea    0x90(%rsp),%rsi
  402b86:	00 
  402b87:	48 8d 7c 24 70       	lea    0x70(%rsp),%rdi
  402b8c:	e8 6f e9 ff ff       	callq  401500 <_ZN2cv12_OutputArrayC1ERNS_3MatE@plt>
  402b91:	48 8d 74 24 70       	lea    0x70(%rsp),%rsi
  402b96:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  402b9b:	e8 70 ea ff ff       	callq  401610 <_ZN2cv5mergeERKSt6vectorINS_3MatESaIS1_EERKNS_12_OutputArrayE@plt>
  402ba0:	48 8d b4 24 90 00 00 	lea    0x90(%rsp),%rsi
  402ba7:	00 
  402ba8:	48 8d 7c 24 70       	lea    0x70(%rsp),%rdi
  402bad:	48 c7 44 24 50 00 00 	movq   $0x0,0x50(%rsp)
  402bb4:	00 00 
  402bb6:	48 c7 44 24 58 00 00 	movq   $0x0,0x58(%rsp)
  402bbd:	00 00 
  402bbf:	48 c7 44 24 60 00 00 	movq   $0x0,0x60(%rsp)
  402bc6:	00 00 
  402bc8:	e8 33 ea ff ff       	callq  401600 <_ZN2cv11_InputArrayC1ERKNS_3MatE@plt>
  402bcd:	48 8d 54 24 0e       	lea    0xe(%rsp),%rdx
  402bd2:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  402bd7:	be 00 4b 40 00       	mov    $0x404b00,%esi
  402bdc:	e8 2f eb ff ff       	callq  401710 <_ZNSsC1EPKcRKSaIcE@plt>
  402be1:	48 8d 54 24 50       	lea    0x50(%rsp),%rdx
  402be6:	48 8d 74 24 70       	lea    0x70(%rsp),%rsi
  402beb:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  402bf0:	e8 cb e9 ff ff       	callq  4015c0 <_ZN2cv7imwriteERKSsRKNS_11_InputArrayERKSt6vectorIiSaIiEE@plt>
  402bf5:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  402bfa:	48 8d 78 e8          	lea    -0x18(%rax),%rdi
  402bfe:	48 81 ff 80 61 60 00 	cmp    $0x606180,%rdi
  402c05:	0f 85 14 04 00 00    	jne    40301f <_Z11encodeImagePfS_S_+0x86f>
  402c0b:	48 8b 7c 24 50       	mov    0x50(%rsp),%rdi
  402c10:	48 85 ff             	test   %rdi,%rdi
  402c13:	74 05                	je     402c1a <_Z11encodeImagePfS_S_+0x46a>
  402c15:	e8 76 e9 ff ff       	callq  401590 <_ZdlPv@plt>
  402c1a:	48 8b 84 24 c8 01 00 	mov    0x1c8(%rsp),%rax
  402c21:	00 
  402c22:	48 85 c0             	test   %rax,%rax
  402c25:	74 0a                	je     402c31 <_Z11encodeImagePfS_S_+0x481>
  402c27:	f0 83 28 01          	lock subl $0x1,(%rax)
  402c2b:	0f 84 cf 02 00 00    	je     402f00 <_Z11encodeImagePfS_S_+0x750>
  402c31:	48 8b 84 24 f0 01 00 	mov    0x1f0(%rsp),%rax
  402c38:	00 
  402c39:	48 8b bc 24 f8 01 00 	mov    0x1f8(%rsp),%rdi
  402c40:	00 
  402c41:	48 c7 84 24 e0 01 00 	movq   $0x0,0x1e0(%rsp)
  402c48:	00 00 00 00 00 
  402c4d:	48 c7 84 24 d8 01 00 	movq   $0x0,0x1d8(%rsp)
  402c54:	00 00 00 00 00 
  402c59:	48 c7 84 24 d0 01 00 	movq   $0x0,0x1d0(%rsp)
  402c60:	00 00 00 00 00 
  402c65:	48 c7 84 24 c0 01 00 	movq   $0x0,0x1c0(%rsp)
  402c6c:	00 00 00 00 00 
  402c71:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402c77:	48 8d 84 24 00 02 00 	lea    0x200(%rsp),%rax
  402c7e:	00 
  402c7f:	48 c7 84 24 c8 01 00 	movq   $0x0,0x1c8(%rsp)
  402c86:	00 00 00 00 00 
  402c8b:	48 39 c7             	cmp    %rax,%rdi
  402c8e:	74 05                	je     402c95 <_Z11encodeImagePfS_S_+0x4e5>
  402c90:	e8 fb e9 ff ff       	callq  401690 <_ZN2cv8fastFreeEPv@plt>
  402c95:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  402c9c:	00 
  402c9d:	48 85 c0             	test   %rax,%rax
  402ca0:	74 0a                	je     402cac <_Z11encodeImagePfS_S_+0x4fc>
  402ca2:	f0 83 28 01          	lock subl $0x1,(%rax)
  402ca6:	0f 84 8a 02 00 00    	je     402f36 <_Z11encodeImagePfS_S_+0x786>
  402cac:	48 8b 84 24 90 01 00 	mov    0x190(%rsp),%rax
  402cb3:	00 
  402cb4:	48 8b bc 24 98 01 00 	mov    0x198(%rsp),%rdi
  402cbb:	00 
  402cbc:	48 c7 84 24 80 01 00 	movq   $0x0,0x180(%rsp)
  402cc3:	00 00 00 00 00 
  402cc8:	48 c7 84 24 78 01 00 	movq   $0x0,0x178(%rsp)
  402ccf:	00 00 00 00 00 
  402cd4:	48 c7 84 24 70 01 00 	movq   $0x0,0x170(%rsp)
  402cdb:	00 00 00 00 00 
  402ce0:	48 c7 84 24 60 01 00 	movq   $0x0,0x160(%rsp)
  402ce7:	00 00 00 00 00 
  402cec:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402cf2:	48 8d 84 24 a0 01 00 	lea    0x1a0(%rsp),%rax
  402cf9:	00 
  402cfa:	48 c7 84 24 68 01 00 	movq   $0x0,0x168(%rsp)
  402d01:	00 00 00 00 00 
  402d06:	48 39 c7             	cmp    %rax,%rdi
  402d09:	74 05                	je     402d10 <_Z11encodeImagePfS_S_+0x560>
  402d0b:	e8 80 e9 ff ff       	callq  401690 <_ZN2cv8fastFreeEPv@plt>
  402d10:	48 8b 84 24 08 01 00 	mov    0x108(%rsp),%rax
  402d17:	00 
  402d18:	48 85 c0             	test   %rax,%rax
  402d1b:	74 0a                	je     402d27 <_Z11encodeImagePfS_S_+0x577>
  402d1d:	f0 83 28 01          	lock subl $0x1,(%rax)
  402d21:	0f 84 fd 01 00 00    	je     402f24 <_Z11encodeImagePfS_S_+0x774>
  402d27:	48 8b 84 24 30 01 00 	mov    0x130(%rsp),%rax
  402d2e:	00 
  402d2f:	48 8b bc 24 38 01 00 	mov    0x138(%rsp),%rdi
  402d36:	00 
  402d37:	48 c7 84 24 20 01 00 	movq   $0x0,0x120(%rsp)
  402d3e:	00 00 00 00 00 
  402d43:	48 c7 84 24 18 01 00 	movq   $0x0,0x118(%rsp)
  402d4a:	00 00 00 00 00 
  402d4f:	48 c7 84 24 10 01 00 	movq   $0x0,0x110(%rsp)
  402d56:	00 00 00 00 00 
  402d5b:	48 c7 84 24 00 01 00 	movq   $0x0,0x100(%rsp)
  402d62:	00 00 00 00 00 
  402d67:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402d6d:	48 8d 84 24 40 01 00 	lea    0x140(%rsp),%rax
  402d74:	00 
  402d75:	48 c7 84 24 08 01 00 	movq   $0x0,0x108(%rsp)
  402d7c:	00 00 00 00 00 
  402d81:	48 39 c7             	cmp    %rax,%rdi
  402d84:	74 05                	je     402d8b <_Z11encodeImagePfS_S_+0x5db>
  402d86:	e8 05 e9 ff ff       	callq  401690 <_ZN2cv8fastFreeEPv@plt>
  402d8b:	48 8b 84 24 a8 00 00 	mov    0xa8(%rsp),%rax
  402d92:	00 
  402d93:	48 85 c0             	test   %rax,%rax
  402d96:	74 0a                	je     402da2 <_Z11encodeImagePfS_S_+0x5f2>
  402d98:	f0 83 28 01          	lock subl $0x1,(%rax)
  402d9c:	0f 84 70 01 00 00    	je     402f12 <_Z11encodeImagePfS_S_+0x762>
  402da2:	48 8b 84 24 d0 00 00 	mov    0xd0(%rsp),%rax
  402da9:	00 
  402daa:	48 8b bc 24 d8 00 00 	mov    0xd8(%rsp),%rdi
  402db1:	00 
  402db2:	48 c7 84 24 c0 00 00 	movq   $0x0,0xc0(%rsp)
  402db9:	00 00 00 00 00 
  402dbe:	48 c7 84 24 b8 00 00 	movq   $0x0,0xb8(%rsp)
  402dc5:	00 00 00 00 00 
  402dca:	48 c7 84 24 b0 00 00 	movq   $0x0,0xb0(%rsp)
  402dd1:	00 00 00 00 00 
  402dd6:	48 c7 84 24 a0 00 00 	movq   $0x0,0xa0(%rsp)
  402ddd:	00 00 00 00 00 
  402de2:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402de8:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
  402def:	00 
  402df0:	48 c7 84 24 a8 00 00 	movq   $0x0,0xa8(%rsp)
  402df7:	00 00 00 00 00 
  402dfc:	48 39 c7             	cmp    %rax,%rdi
  402dff:	74 05                	je     402e06 <_Z11encodeImagePfS_S_+0x656>
  402e01:	e8 8a e8 ff ff       	callq  401690 <_ZN2cv8fastFreeEPv@plt>
  402e06:	4c 8b 64 24 38       	mov    0x38(%rsp),%r12
  402e0b:	48 8b 5c 24 30       	mov    0x30(%rsp),%rbx
  402e10:	49 39 dc             	cmp    %rbx,%r12
  402e13:	4c 89 e7             	mov    %r12,%rdi
  402e16:	75 55                	jne    402e6d <_Z11encodeImagePfS_S_+0x6bd>
  402e18:	e9 a0 00 00 00       	jmpq   402ebd <_Z11encodeImagePfS_S_+0x70d>
  402e1d:	0f 1f 00             	nopl   (%rax)
  402e20:	48 8b 43 40          	mov    0x40(%rbx),%rax
  402e24:	48 c7 43 30 00 00 00 	movq   $0x0,0x30(%rbx)
  402e2b:	00 
  402e2c:	48 c7 43 28 00 00 00 	movq   $0x0,0x28(%rbx)
  402e33:	00 
  402e34:	48 c7 43 20 00 00 00 	movq   $0x0,0x20(%rbx)
  402e3b:	00 
  402e3c:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
  402e43:	00 
  402e44:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402e4a:	48 8b 7b 48          	mov    0x48(%rbx),%rdi
  402e4e:	48 8d 43 50          	lea    0x50(%rbx),%rax
  402e52:	48 c7 43 18 00 00 00 	movq   $0x0,0x18(%rbx)
  402e59:	00 
  402e5a:	48 39 c7             	cmp    %rax,%rdi
  402e5d:	74 05                	je     402e64 <_Z11encodeImagePfS_S_+0x6b4>
  402e5f:	e8 2c e8 ff ff       	callq  401690 <_ZN2cv8fastFreeEPv@plt>
  402e64:	48 83 c3 60          	add    $0x60,%rbx
  402e68:	49 39 dc             	cmp    %rbx,%r12
  402e6b:	74 4b                	je     402eb8 <_Z11encodeImagePfS_S_+0x708>
  402e6d:	48 8b 43 18          	mov    0x18(%rbx),%rax
  402e71:	48 85 c0             	test   %rax,%rax
  402e74:	74 aa                	je     402e20 <_Z11encodeImagePfS_S_+0x670>
  402e76:	f0 83 28 01          	lock subl $0x1,(%rax)
  402e7a:	75 a4                	jne    402e20 <_Z11encodeImagePfS_S_+0x670>
  402e7c:	48 89 df             	mov    %rbx,%rdi
  402e7f:	e8 fc e6 ff ff       	callq  401580 <_ZN2cv3Mat10deallocateEv@plt>
  402e84:	eb 9a                	jmp    402e20 <_Z11encodeImagePfS_S_+0x670>
  402e86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  402e8d:	00 00 00 
  402e90:	48 8d b4 24 50 01 00 	lea    0x150(%rsp),%rsi
  402e97:	00 
  402e98:	c7 47 04 00 00 00 00 	movl   $0x0,0x4(%rdi)
  402e9f:	e8 1c e6 ff ff       	callq  4014c0 <_ZN2cv3Mat8copySizeERKS0_@plt>
  402ea4:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  402ea9:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  402eae:	e9 f8 fb ff ff       	jmpq   402aab <_Z11encodeImagePfS_S_+0x2fb>
  402eb3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  402eb8:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  402ebd:	48 85 ff             	test   %rdi,%rdi
  402ec0:	74 05                	je     402ec7 <_Z11encodeImagePfS_S_+0x717>
  402ec2:	e8 c9 e6 ff ff       	callq  401590 <_ZdlPv@plt>
  402ec7:	48 8d 65 f0          	lea    -0x10(%rbp),%rsp
  402ecb:	31 c0                	xor    %eax,%eax
  402ecd:	5b                   	pop    %rbx
  402ece:	41 5c                	pop    %r12
  402ed0:	5d                   	pop    %rbp
  402ed1:	c3                   	retq   
  402ed2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  402ed8:	48 8d b4 24 f0 00 00 	lea    0xf0(%rsp),%rsi
  402edf:	00 
  402ee0:	c7 47 04 00 00 00 00 	movl   $0x0,0x4(%rdi)
  402ee7:	e8 d4 e5 ff ff       	callq  4014c0 <_ZN2cv3Mat8copySizeERKS0_@plt>
  402eec:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  402ef1:	e9 80 fc ff ff       	jmpq   402b76 <_Z11encodeImagePfS_S_+0x3c6>
  402ef6:	48 89 c6             	mov    %rax,%rsi
  402ef9:	31 ff                	xor    %edi,%edi
  402efb:	e9 ab fb ff ff       	jmpq   402aab <_Z11encodeImagePfS_S_+0x2fb>
  402f00:	48 8d bc 24 b0 01 00 	lea    0x1b0(%rsp),%rdi
  402f07:	00 
  402f08:	e8 73 e6 ff ff       	callq  401580 <_ZN2cv3Mat10deallocateEv@plt>
  402f0d:	e9 1f fd ff ff       	jmpq   402c31 <_Z11encodeImagePfS_S_+0x481>
  402f12:	48 8d bc 24 90 00 00 	lea    0x90(%rsp),%rdi
  402f19:	00 
  402f1a:	e8 61 e6 ff ff       	callq  401580 <_ZN2cv3Mat10deallocateEv@plt>
  402f1f:	e9 7e fe ff ff       	jmpq   402da2 <_Z11encodeImagePfS_S_+0x5f2>
  402f24:	48 8d bc 24 f0 00 00 	lea    0xf0(%rsp),%rdi
  402f2b:	00 
  402f2c:	e8 4f e6 ff ff       	callq  401580 <_ZN2cv3Mat10deallocateEv@plt>
  402f31:	e9 f1 fd ff ff       	jmpq   402d27 <_Z11encodeImagePfS_S_+0x577>
  402f36:	48 8d bc 24 50 01 00 	lea    0x150(%rsp),%rdi
  402f3d:	00 
  402f3e:	e8 3d e6 ff ff       	callq  401580 <_ZN2cv3Mat10deallocateEv@plt>
  402f43:	e9 64 fd ff ff       	jmpq   402cac <_Z11encodeImagePfS_S_+0x4fc>
  402f48:	48 8d 94 24 f0 00 00 	lea    0xf0(%rsp),%rdx
  402f4f:	00 
  402f50:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  402f55:	e8 56 0b 00 00       	callq  403ab0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_>
  402f5a:	e9 20 fc ff ff       	jmpq   402b7f <_Z11encodeImagePfS_S_+0x3cf>
  402f5f:	48 89 fe             	mov    %rdi,%rsi
  402f62:	48 8d 94 24 50 01 00 	lea    0x150(%rsp),%rdx
  402f69:	00 
  402f6a:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  402f6f:	e8 3c 0b 00 00       	callq  403ab0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_>
  402f74:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  402f79:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  402f7e:	e9 31 fb ff ff       	jmpq   402ab4 <_Z11encodeImagePfS_S_+0x304>
  402f83:	48 89 c3             	mov    %rax,%rbx
  402f86:	c5 f8 77             	vzeroupper 
  402f89:	48 8d bc 24 b0 01 00 	lea    0x1b0(%rsp),%rdi
  402f90:	00 
  402f91:	e8 ba 09 00 00       	callq  403950 <_ZN2cv3MatD1Ev>
  402f96:	48 8d bc 24 50 01 00 	lea    0x150(%rsp),%rdi
  402f9d:	00 
  402f9e:	e8 ad 09 00 00       	callq  403950 <_ZN2cv3MatD1Ev>
  402fa3:	48 8d bc 24 f0 00 00 	lea    0xf0(%rsp),%rdi
  402faa:	00 
  402fab:	e8 a0 09 00 00       	callq  403950 <_ZN2cv3MatD1Ev>
  402fb0:	48 8d bc 24 90 00 00 	lea    0x90(%rsp),%rdi
  402fb7:	00 
  402fb8:	e8 93 09 00 00       	callq  403950 <_ZN2cv3MatD1Ev>
  402fbd:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  402fc2:	e8 09 0a 00 00       	callq  4039d0 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev>
  402fc7:	48 89 df             	mov    %rbx,%rdi
  402fca:	e8 11 e6 ff ff       	callq  4015e0 <_Unwind_Resume@plt>
  402fcf:	48 89 c3             	mov    %rax,%rbx
  402fd2:	c5 f8 77             	vzeroupper 
  402fd5:	eb e6                	jmp    402fbd <_Z11encodeImagePfS_S_+0x80d>
  402fd7:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  402fdc:	48 89 c3             	mov    %rax,%rbx
  402fdf:	48 85 ff             	test   %rdi,%rdi
  402fe2:	74 69                	je     40304d <_Z11encodeImagePfS_S_+0x89d>
  402fe4:	c5 f8 77             	vzeroupper 
  402fe7:	e8 a4 e5 ff ff       	callq  401590 <_ZdlPv@plt>
  402fec:	eb d9                	jmp    402fc7 <_Z11encodeImagePfS_S_+0x817>
  402fee:	48 89 c3             	mov    %rax,%rbx
  402ff1:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  402ff6:	48 8d 74 24 0f       	lea    0xf(%rsp),%rsi
  402ffb:	48 8d 78 e8          	lea    -0x18(%rax),%rdi
  402fff:	c5 f8 77             	vzeroupper 
  403002:	e8 39 e6 ff ff       	callq  401640 <_ZNSs4_Rep10_M_disposeERKSaIcE@plt>
  403007:	48 8b 7c 24 50       	mov    0x50(%rsp),%rdi
  40300c:	48 85 ff             	test   %rdi,%rdi
  40300f:	0f 84 74 ff ff ff    	je     402f89 <_Z11encodeImagePfS_S_+0x7d9>
  403015:	e8 76 e5 ff ff       	callq  401590 <_ZdlPv@plt>
  40301a:	e9 6a ff ff ff       	jmpq   402f89 <_Z11encodeImagePfS_S_+0x7d9>
  40301f:	b9 a0 16 40 00       	mov    $0x4016a0,%ecx
  403024:	48 8d 57 10          	lea    0x10(%rdi),%rdx
  403028:	48 85 c9             	test   %rcx,%rcx
  40302b:	74 28                	je     403055 <_Z11encodeImagePfS_S_+0x8a5>
  40302d:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  403032:	f0 0f c1 02          	lock xadd %eax,(%rdx)
  403036:	85 c0                	test   %eax,%eax
  403038:	0f 8f cd fb ff ff    	jg     402c0b <_Z11encodeImagePfS_S_+0x45b>
  40303e:	48 8d 74 24 0f       	lea    0xf(%rsp),%rsi
  403043:	e8 78 e6 ff ff       	callq  4016c0 <_ZNSs4_Rep10_M_destroyERKSaIcE@plt>
  403048:	e9 be fb ff ff       	jmpq   402c0b <_Z11encodeImagePfS_S_+0x45b>
  40304d:	c5 f8 77             	vzeroupper 
  403050:	e9 72 ff ff ff       	jmpq   402fc7 <_Z11encodeImagePfS_S_+0x817>
  403055:	8b 50 f8             	mov    -0x8(%rax),%edx
  403058:	8d 4a ff             	lea    -0x1(%rdx),%ecx
  40305b:	89 48 f8             	mov    %ecx,-0x8(%rax)
  40305e:	89 d0                	mov    %edx,%eax
  403060:	eb d4                	jmp    403036 <_Z11encodeImagePfS_S_+0x886>
  403062:	48 89 c3             	mov    %rax,%rbx
  403065:	c5 f8 77             	vzeroupper 
  403068:	eb 9d                	jmp    403007 <_Z11encodeImagePfS_S_+0x857>
  40306a:	48 89 c3             	mov    %rax,%rbx
  40306d:	c5 f8 77             	vzeroupper 
  403070:	e9 21 ff ff ff       	jmpq   402f96 <_Z11encodeImagePfS_S_+0x7e6>
  403075:	48 89 c3             	mov    %rax,%rbx
  403078:	c5 f8 77             	vzeroupper 
  40307b:	e9 23 ff ff ff       	jmpq   402fa3 <_Z11encodeImagePfS_S_+0x7f3>
  403080:	48 89 c3             	mov    %rax,%rbx
  403083:	c5 f8 77             	vzeroupper 
  403086:	e9 25 ff ff ff       	jmpq   402fb0 <_Z11encodeImagePfS_S_+0x800>
  40308b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000403090 <_Z11decodeImagePfS_S_>:
  403090:	55                   	push   %rbp
  403091:	48 89 e5             	mov    %rsp,%rbp
  403094:	41 57                	push   %r15
  403096:	41 56                	push   %r14
  403098:	41 55                	push   %r13
  40309a:	49 89 d5             	mov    %rdx,%r13
  40309d:	41 54                	push   %r12
  40309f:	49 89 f4             	mov    %rsi,%r12
  4030a2:	be 48 4b 40 00       	mov    $0x404b48,%esi
  4030a7:	53                   	push   %rbx
  4030a8:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4030ac:	48 81 ec a0 02 00 00 	sub    $0x2a0,%rsp
  4030b3:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
  4030b8:	48 8d 9c 24 80 01 00 	lea    0x180(%rsp),%rbx
  4030bf:	00 
  4030c0:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4030c5:	48 c7 84 24 b8 00 00 	movq   $0x0,0xb8(%rsp)
  4030cc:	00 00 00 00 00 
  4030d1:	48 c7 84 24 b0 00 00 	movq   $0x0,0xb0(%rsp)
  4030d8:	00 00 00 00 00 
  4030dd:	48 8d 54 24 1e       	lea    0x1e(%rsp),%rdx
  4030e2:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
  4030e9:	00 
  4030ea:	48 8d 84 24 b0 00 00 	lea    0xb0(%rsp),%rax
  4030f1:	00 
  4030f2:	c7 44 24 60 00 00 ff 	movl   $0x42ff0000,0x60(%rsp)
  4030f9:	42 
  4030fa:	c7 44 24 6c 00 00 00 	movl   $0x0,0x6c(%rsp)
  403101:	00 
  403102:	c7 44 24 68 00 00 00 	movl   $0x0,0x68(%rsp)
  403109:	00 
  40310a:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
  40310f:	48 89 84 24 a8 00 00 	mov    %rax,0xa8(%rsp)
  403116:	00 
  403117:	48 8d 84 24 c8 00 00 	lea    0xc8(%rsp),%rax
  40311e:	00 
  40311f:	c7 44 24 64 00 00 00 	movl   $0x0,0x64(%rsp)
  403126:	00 
  403127:	48 c7 84 24 90 00 00 	movq   $0x0,0x90(%rsp)
  40312e:	00 00 00 00 00 
  403133:	48 c7 84 24 88 00 00 	movq   $0x0,0x88(%rsp)
  40313a:	00 00 00 00 00 
  40313f:	48 89 84 24 00 01 00 	mov    %rax,0x100(%rsp)
  403146:	00 
  403147:	48 8d 84 24 10 01 00 	lea    0x110(%rsp),%rax
  40314e:	00 
  40314f:	48 c7 84 24 80 00 00 	movq   $0x0,0x80(%rsp)
  403156:	00 00 00 00 00 
  40315b:	48 c7 44 24 70 00 00 	movq   $0x0,0x70(%rsp)
  403162:	00 00 
  403164:	48 c7 44 24 78 00 00 	movq   $0x0,0x78(%rsp)
  40316b:	00 00 
  40316d:	48 89 84 24 08 01 00 	mov    %rax,0x108(%rsp)
  403174:	00 
  403175:	48 8d 43 08          	lea    0x8(%rbx),%rax
  403179:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
  403180:	00 00 00 00 00 
  403185:	48 c7 84 24 18 01 00 	movq   $0x0,0x118(%rsp)
  40318c:	00 00 00 00 00 
  403191:	48 c7 84 24 10 01 00 	movq   $0x0,0x110(%rsp)
  403198:	00 00 00 00 00 
  40319d:	c7 84 24 c0 00 00 00 	movl   $0x42ff0000,0xc0(%rsp)
  4031a4:	00 00 ff 42 
  4031a8:	c7 84 24 cc 00 00 00 	movl   $0x0,0xcc(%rsp)
  4031af:	00 00 00 00 
  4031b3:	c7 84 24 c8 00 00 00 	movl   $0x0,0xc8(%rsp)
  4031ba:	00 00 00 00 
  4031be:	c7 84 24 c4 00 00 00 	movl   $0x0,0xc4(%rsp)
  4031c5:	00 00 00 00 
  4031c9:	48 c7 84 24 f0 00 00 	movq   $0x0,0xf0(%rsp)
  4031d0:	00 00 00 00 00 
  4031d5:	48 c7 84 24 e8 00 00 	movq   $0x0,0xe8(%rsp)
  4031dc:	00 00 00 00 00 
  4031e1:	48 c7 84 24 e0 00 00 	movq   $0x0,0xe0(%rsp)
  4031e8:	00 00 00 00 00 
  4031ed:	48 c7 84 24 d0 00 00 	movq   $0x0,0xd0(%rsp)
  4031f4:	00 00 00 00 00 
  4031f9:	48 89 84 24 c0 01 00 	mov    %rax,0x1c0(%rsp)
  403200:	00 
  403201:	48 8d 43 50          	lea    0x50(%rbx),%rax
  403205:	48 c7 84 24 d8 00 00 	movq   $0x0,0xd8(%rsp)
  40320c:	00 00 00 00 00 
  403211:	48 c7 84 24 f8 00 00 	movq   $0x0,0xf8(%rsp)
  403218:	00 00 00 00 00 
  40321d:	48 c7 84 24 d8 01 00 	movq   $0x0,0x1d8(%rsp)
  403224:	00 00 00 00 00 
  403229:	48 89 84 24 c8 01 00 	mov    %rax,0x1c8(%rsp)
  403230:	00 
  403231:	48 8d 43 68          	lea    0x68(%rbx),%rax
  403235:	48 c7 84 24 d0 01 00 	movq   $0x0,0x1d0(%rsp)
  40323c:	00 00 00 00 00 
  403241:	c7 84 24 80 01 00 00 	movl   $0x42ff0000,0x180(%rsp)
  403248:	00 00 ff 42 
  40324c:	c7 84 24 8c 01 00 00 	movl   $0x0,0x18c(%rsp)
  403253:	00 00 00 00 
  403257:	48 89 84 24 20 02 00 	mov    %rax,0x220(%rsp)
  40325e:	00 
  40325f:	48 8d 83 b0 00 00 00 	lea    0xb0(%rbx),%rax
  403266:	c7 84 24 88 01 00 00 	movl   $0x0,0x188(%rsp)
  40326d:	00 00 00 00 
  403271:	c7 84 24 84 01 00 00 	movl   $0x0,0x184(%rsp)
  403278:	00 00 00 00 
  40327c:	48 c7 84 24 b0 01 00 	movq   $0x0,0x1b0(%rsp)
  403283:	00 00 00 00 00 
  403288:	48 89 84 24 28 02 00 	mov    %rax,0x228(%rsp)
  40328f:	00 
  403290:	48 8d 83 c8 00 00 00 	lea    0xc8(%rbx),%rax
  403297:	48 c7 84 24 a8 01 00 	movq   $0x0,0x1a8(%rsp)
  40329e:	00 00 00 00 00 
  4032a3:	48 c7 84 24 a0 01 00 	movq   $0x0,0x1a0(%rsp)
  4032aa:	00 00 00 00 00 
  4032af:	48 c7 84 24 90 01 00 	movq   $0x0,0x190(%rsp)
  4032b6:	00 00 00 00 00 
  4032bb:	48 89 84 24 80 02 00 	mov    %rax,0x280(%rsp)
  4032c2:	00 
  4032c3:	48 8d 83 10 01 00 00 	lea    0x110(%rbx),%rax
  4032ca:	48 c7 84 24 98 01 00 	movq   $0x0,0x198(%rsp)
  4032d1:	00 00 00 00 00 
  4032d6:	48 c7 84 24 b8 01 00 	movq   $0x0,0x1b8(%rsp)
  4032dd:	00 00 00 00 00 
  4032e2:	48 c7 84 24 38 02 00 	movq   $0x0,0x238(%rsp)
  4032e9:	00 00 00 00 00 
  4032ee:	48 c7 84 24 30 02 00 	movq   $0x0,0x230(%rsp)
  4032f5:	00 00 00 00 00 
  4032fa:	c7 84 24 e0 01 00 00 	movl   $0x42ff0000,0x1e0(%rsp)
  403301:	00 00 ff 42 
  403305:	c7 84 24 ec 01 00 00 	movl   $0x0,0x1ec(%rsp)
  40330c:	00 00 00 00 
  403310:	c7 84 24 e8 01 00 00 	movl   $0x0,0x1e8(%rsp)
  403317:	00 00 00 00 
  40331b:	c7 84 24 e4 01 00 00 	movl   $0x0,0x1e4(%rsp)
  403322:	00 00 00 00 
  403326:	48 c7 84 24 10 02 00 	movq   $0x0,0x210(%rsp)
  40332d:	00 00 00 00 00 
  403332:	48 c7 84 24 08 02 00 	movq   $0x0,0x208(%rsp)
  403339:	00 00 00 00 00 
  40333e:	48 c7 84 24 00 02 00 	movq   $0x0,0x200(%rsp)
  403345:	00 00 00 00 00 
  40334a:	48 c7 84 24 f0 01 00 	movq   $0x0,0x1f0(%rsp)
  403351:	00 00 00 00 00 
  403356:	48 c7 84 24 f8 01 00 	movq   $0x0,0x1f8(%rsp)
  40335d:	00 00 00 00 00 
  403362:	48 c7 84 24 18 02 00 	movq   $0x0,0x218(%rsp)
  403369:	00 00 00 00 00 
  40336e:	48 89 84 24 88 02 00 	mov    %rax,0x288(%rsp)
  403375:	00 
  403376:	48 c7 84 24 98 02 00 	movq   $0x0,0x298(%rsp)
  40337d:	00 00 00 00 00 
  403382:	48 c7 84 24 90 02 00 	movq   $0x0,0x290(%rsp)
  403389:	00 00 00 00 00 
  40338e:	c7 84 24 40 02 00 00 	movl   $0x42ff0000,0x240(%rsp)
  403395:	00 00 ff 42 
  403399:	c7 84 24 4c 02 00 00 	movl   $0x0,0x24c(%rsp)
  4033a0:	00 00 00 00 
  4033a4:	c7 84 24 48 02 00 00 	movl   $0x0,0x248(%rsp)
  4033ab:	00 00 00 00 
  4033af:	c7 84 24 44 02 00 00 	movl   $0x0,0x244(%rsp)
  4033b6:	00 00 00 00 
  4033ba:	48 c7 84 24 70 02 00 	movq   $0x0,0x270(%rsp)
  4033c1:	00 00 00 00 00 
  4033c6:	48 c7 84 24 68 02 00 	movq   $0x0,0x268(%rsp)
  4033cd:	00 00 00 00 00 
  4033d2:	48 c7 84 24 60 02 00 	movq   $0x0,0x260(%rsp)
  4033d9:	00 00 00 00 00 
  4033de:	48 c7 84 24 50 02 00 	movq   $0x0,0x250(%rsp)
  4033e5:	00 00 00 00 00 
  4033ea:	48 c7 84 24 58 02 00 	movq   $0x0,0x258(%rsp)
  4033f1:	00 00 00 00 00 
  4033f6:	48 c7 84 24 78 02 00 	movq   $0x0,0x278(%rsp)
  4033fd:	00 00 00 00 00 
  403402:	48 c7 44 24 40 00 00 	movq   $0x0,0x40(%rsp)
  403409:	00 00 
  40340b:	48 c7 44 24 48 00 00 	movq   $0x0,0x48(%rsp)
  403412:	00 00 
  403414:	48 c7 44 24 50 00 00 	movq   $0x0,0x50(%rsp)
  40341b:	00 00 
  40341d:	e8 ee e2 ff ff       	callq  401710 <_ZNSsC1EPKcRKSaIcE@plt>
  403422:	48 8d 74 24 20       	lea    0x20(%rsp),%rsi
  403427:	48 8d bc 24 20 01 00 	lea    0x120(%rsp),%rdi
  40342e:	00 
  40342f:	ba 01 00 00 00       	mov    $0x1,%edx
  403434:	e8 c7 e2 ff ff       	callq  401700 <_ZN2cv6imreadERKSsi@plt>
  403439:	48 8b 84 24 38 01 00 	mov    0x138(%rsp),%rax
  403440:	00 
  403441:	48 85 c0             	test   %rax,%rax
  403444:	74 04                	je     40344a <_Z11decodeImagePfS_S_+0x3ba>
  403446:	f0 83 00 01          	lock addl $0x1,(%rax)
  40344a:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  40344f:	48 85 c0             	test   %rax,%rax
  403452:	74 0a                	je     40345e <_Z11decodeImagePfS_S_+0x3ce>
  403454:	f0 83 28 01          	lock subl $0x1,(%rax)
  403458:	0f 84 d2 03 00 00    	je     403830 <_Z11decodeImagePfS_S_+0x7a0>
  40345e:	48 8b 84 24 a0 00 00 	mov    0xa0(%rsp),%rax
  403465:	00 
  403466:	48 c7 84 24 90 00 00 	movq   $0x0,0x90(%rsp)
  40346d:	00 00 00 00 00 
  403472:	48 c7 84 24 88 00 00 	movq   $0x0,0x88(%rsp)
  403479:	00 00 00 00 00 
  40347e:	48 c7 84 24 80 00 00 	movq   $0x0,0x80(%rsp)
  403485:	00 00 00 00 00 
  40348a:	48 c7 44 24 70 00 00 	movq   $0x0,0x70(%rsp)
  403491:	00 00 
  403493:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  403499:	83 7c 24 64 02       	cmpl   $0x2,0x64(%rsp)
  40349e:	8b 84 24 20 01 00 00 	mov    0x120(%rsp),%eax
  4034a5:	48 c7 44 24 78 00 00 	movq   $0x0,0x78(%rsp)
  4034ac:	00 00 
  4034ae:	89 44 24 60          	mov    %eax,0x60(%rsp)
  4034b2:	7f 10                	jg     4034c4 <_Z11decodeImagePfS_S_+0x434>
  4034b4:	8b 84 24 24 01 00 00 	mov    0x124(%rsp),%eax
  4034bb:	83 f8 02             	cmp    $0x2,%eax
  4034be:	0f 8e 1c 03 00 00    	jle    4037e0 <_Z11decodeImagePfS_S_+0x750>
  4034c4:	48 8d b4 24 20 01 00 	lea    0x120(%rsp),%rsi
  4034cb:	00 
  4034cc:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  4034d1:	e8 ea df ff ff       	callq  4014c0 <_ZN2cv3Mat8copySizeERKS0_@plt>
  4034d6:	c5 f9 6f 84 24 30 01 	vmovdqa 0x130(%rsp),%xmm0
  4034dd:	00 00 
  4034df:	48 8b 84 24 38 01 00 	mov    0x138(%rsp),%rax
  4034e6:	00 
  4034e7:	c5 f9 7f 44 24 70    	vmovdqa %xmm0,0x70(%rsp)
  4034ed:	48 85 c0             	test   %rax,%rax
  4034f0:	c5 f9 6f 84 24 40 01 	vmovdqa 0x140(%rsp),%xmm0
  4034f7:	00 00 
  4034f9:	c5 f9 7f 84 24 80 00 	vmovdqa %xmm0,0x80(%rsp)
  403500:	00 00 
  403502:	c5 f9 6f 84 24 50 01 	vmovdqa 0x150(%rsp),%xmm0
  403509:	00 00 
  40350b:	c5 f9 7f 84 24 90 00 	vmovdqa %xmm0,0x90(%rsp)
  403512:	00 00 
  403514:	74 0a                	je     403520 <_Z11decodeImagePfS_S_+0x490>
  403516:	f0 83 28 01          	lock subl $0x1,(%rax)
  40351a:	0f 84 20 03 00 00    	je     403840 <_Z11decodeImagePfS_S_+0x7b0>
  403520:	48 8b 84 24 60 01 00 	mov    0x160(%rsp),%rax
  403527:	00 
  403528:	48 8b bc 24 68 01 00 	mov    0x168(%rsp),%rdi
  40352f:	00 
  403530:	48 c7 84 24 50 01 00 	movq   $0x0,0x150(%rsp)
  403537:	00 00 00 00 00 
  40353c:	48 c7 84 24 48 01 00 	movq   $0x0,0x148(%rsp)
  403543:	00 00 00 00 00 
  403548:	48 c7 84 24 40 01 00 	movq   $0x0,0x140(%rsp)
  40354f:	00 00 00 00 00 
  403554:	48 c7 84 24 30 01 00 	movq   $0x0,0x130(%rsp)
  40355b:	00 00 00 00 00 
  403560:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  403566:	48 8d 84 24 70 01 00 	lea    0x170(%rsp),%rax
  40356d:	00 
  40356e:	48 c7 84 24 38 01 00 	movq   $0x0,0x138(%rsp)
  403575:	00 00 00 00 00 
  40357a:	48 39 c7             	cmp    %rax,%rdi
  40357d:	74 05                	je     403584 <_Z11decodeImagePfS_S_+0x4f4>
  40357f:	e8 0c e1 ff ff       	callq  401690 <_ZN2cv8fastFreeEPv@plt>
  403584:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  403589:	48 8d 78 e8          	lea    -0x18(%rax),%rdi
  40358d:	48 81 ff 80 61 60 00 	cmp    $0x606180,%rdi
  403594:	0f 85 e5 02 00 00    	jne    40387f <_Z11decodeImagePfS_S_+0x7ef>
  40359a:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  40359f:	48 89 de             	mov    %rbx,%rsi
  4035a2:	44 8b 74 24 68       	mov    0x68(%rsp),%r14d
  4035a7:	44 8b 7c 24 6c       	mov    0x6c(%rsp),%r15d
  4035ac:	e8 3f e1 ff ff       	callq  4016f0 <_ZN2cv5splitERKNS_3MatEPS0_@plt>
  4035b1:	48 8b 94 24 a8 01 00 	mov    0x1a8(%rsp),%rdx
  4035b8:	00 
  4035b9:	48 8b b4 24 a0 01 00 	mov    0x1a0(%rsp),%rsi
  4035c0:	00 
  4035c1:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  4035c6:	c6 04 24 00          	movb   $0x0,(%rsp)
  4035ca:	e8 91 0e 00 00       	callq  404460 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag>
  4035cf:	45 0f af f7          	imul   %r15d,%r14d
  4035d3:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  4035d8:	4c 89 ef             	mov    %r13,%rdi
  4035db:	4d 63 f6             	movslq %r14d,%r14
  4035de:	49 c1 e6 02          	shl    $0x2,%r14
  4035e2:	4c 89 f2             	mov    %r14,%rdx
  4035e5:	e8 c6 de ff ff       	callq  4014b0 <memcpy@plt>
  4035ea:	48 8b 94 24 08 02 00 	mov    0x208(%rsp),%rdx
  4035f1:	00 
  4035f2:	48 8b b4 24 00 02 00 	mov    0x200(%rsp),%rsi
  4035f9:	00 
  4035fa:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  4035ff:	c6 04 24 00          	movb   $0x0,(%rsp)
  403603:	e8 58 0e 00 00       	callq  404460 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag>
  403608:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  40360d:	4c 89 f2             	mov    %r14,%rdx
  403610:	4c 89 e7             	mov    %r12,%rdi
  403613:	e8 98 de ff ff       	callq  4014b0 <memcpy@plt>
  403618:	48 8b 94 24 68 02 00 	mov    0x268(%rsp),%rdx
  40361f:	00 
  403620:	48 8b b4 24 60 02 00 	mov    0x260(%rsp),%rsi
  403627:	00 
  403628:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  40362d:	c6 04 24 00          	movb   $0x0,(%rsp)
  403631:	e8 2a 0e 00 00       	callq  404460 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag>
  403636:	4c 8b 64 24 40       	mov    0x40(%rsp),%r12
  40363b:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  403640:	4c 89 f2             	mov    %r14,%rdx
  403643:	4c 89 e6             	mov    %r12,%rsi
  403646:	e8 65 de ff ff       	callq  4014b0 <memcpy@plt>
  40364b:	4d 85 e4             	test   %r12,%r12
  40364e:	74 08                	je     403658 <_Z11decodeImagePfS_S_+0x5c8>
  403650:	4c 89 e7             	mov    %r12,%rdi
  403653:	e8 38 df ff ff       	callq  401590 <_ZdlPv@plt>
  403658:	4c 8d a4 24 a0 02 00 	lea    0x2a0(%rsp),%r12
  40365f:	00 
  403660:	49 83 ec 60          	sub    $0x60,%r12
  403664:	49 8b 44 24 18       	mov    0x18(%r12),%rax
  403669:	48 85 c0             	test   %rax,%rax
  40366c:	74 0a                	je     403678 <_Z11decodeImagePfS_S_+0x5e8>
  40366e:	f0 83 28 01          	lock subl $0x1,(%rax)
  403672:	0f 84 58 01 00 00    	je     4037d0 <_Z11decodeImagePfS_S_+0x740>
  403678:	49 8b 44 24 40       	mov    0x40(%r12),%rax
  40367d:	49 c7 44 24 30 00 00 	movq   $0x0,0x30(%r12)
  403684:	00 00 
  403686:	49 c7 44 24 28 00 00 	movq   $0x0,0x28(%r12)
  40368d:	00 00 
  40368f:	49 c7 44 24 20 00 00 	movq   $0x0,0x20(%r12)
  403696:	00 00 
  403698:	49 c7 44 24 10 00 00 	movq   $0x0,0x10(%r12)
  40369f:	00 00 
  4036a1:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4036a7:	49 8b 7c 24 48       	mov    0x48(%r12),%rdi
  4036ac:	49 8d 44 24 50       	lea    0x50(%r12),%rax
  4036b1:	49 c7 44 24 18 00 00 	movq   $0x0,0x18(%r12)
  4036b8:	00 00 
  4036ba:	48 39 c7             	cmp    %rax,%rdi
  4036bd:	74 05                	je     4036c4 <_Z11decodeImagePfS_S_+0x634>
  4036bf:	e8 cc df ff ff       	callq  401690 <_ZN2cv8fastFreeEPv@plt>
  4036c4:	49 39 dc             	cmp    %rbx,%r12
  4036c7:	75 97                	jne    403660 <_Z11decodeImagePfS_S_+0x5d0>
  4036c9:	48 8b 84 24 d8 00 00 	mov    0xd8(%rsp),%rax
  4036d0:	00 
  4036d1:	48 85 c0             	test   %rax,%rax
  4036d4:	74 0a                	je     4036e0 <_Z11decodeImagePfS_S_+0x650>
  4036d6:	f0 83 28 01          	lock subl $0x1,(%rax)
  4036da:	0f 84 78 01 00 00    	je     403858 <_Z11decodeImagePfS_S_+0x7c8>
  4036e0:	48 8b 84 24 00 01 00 	mov    0x100(%rsp),%rax
  4036e7:	00 
  4036e8:	48 8b bc 24 08 01 00 	mov    0x108(%rsp),%rdi
  4036ef:	00 
  4036f0:	48 c7 84 24 f0 00 00 	movq   $0x0,0xf0(%rsp)
  4036f7:	00 00 00 00 00 
  4036fc:	48 c7 84 24 e8 00 00 	movq   $0x0,0xe8(%rsp)
  403703:	00 00 00 00 00 
  403708:	48 c7 84 24 e0 00 00 	movq   $0x0,0xe0(%rsp)
  40370f:	00 00 00 00 00 
  403714:	48 c7 84 24 d0 00 00 	movq   $0x0,0xd0(%rsp)
  40371b:	00 00 00 00 00 
  403720:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  403726:	48 8d 84 24 10 01 00 	lea    0x110(%rsp),%rax
  40372d:	00 
  40372e:	48 c7 84 24 d8 00 00 	movq   $0x0,0xd8(%rsp)
  403735:	00 00 00 00 00 
  40373a:	48 39 c7             	cmp    %rax,%rdi
  40373d:	74 05                	je     403744 <_Z11decodeImagePfS_S_+0x6b4>
  40373f:	e8 4c df ff ff       	callq  401690 <_ZN2cv8fastFreeEPv@plt>
  403744:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  403749:	48 85 c0             	test   %rax,%rax
  40374c:	74 0a                	je     403758 <_Z11decodeImagePfS_S_+0x6c8>
  40374e:	f0 83 28 01          	lock subl $0x1,(%rax)
  403752:	0f 84 18 01 00 00    	je     403870 <_Z11decodeImagePfS_S_+0x7e0>
  403758:	48 8b 84 24 a0 00 00 	mov    0xa0(%rsp),%rax
  40375f:	00 
  403760:	48 8b bc 24 a8 00 00 	mov    0xa8(%rsp),%rdi
  403767:	00 
  403768:	48 c7 84 24 90 00 00 	movq   $0x0,0x90(%rsp)
  40376f:	00 00 00 00 00 
  403774:	48 c7 84 24 88 00 00 	movq   $0x0,0x88(%rsp)
  40377b:	00 00 00 00 00 
  403780:	48 c7 84 24 80 00 00 	movq   $0x0,0x80(%rsp)
  403787:	00 00 00 00 00 
  40378c:	48 c7 44 24 70 00 00 	movq   $0x0,0x70(%rsp)
  403793:	00 00 
  403795:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40379b:	48 8d 84 24 b0 00 00 	lea    0xb0(%rsp),%rax
  4037a2:	00 
  4037a3:	48 c7 44 24 78 00 00 	movq   $0x0,0x78(%rsp)
  4037aa:	00 00 
  4037ac:	48 39 c7             	cmp    %rax,%rdi
  4037af:	74 05                	je     4037b6 <_Z11decodeImagePfS_S_+0x726>
  4037b1:	e8 da de ff ff       	callq  401690 <_ZN2cv8fastFreeEPv@plt>
  4037b6:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
  4037ba:	31 c0                	xor    %eax,%eax
  4037bc:	5b                   	pop    %rbx
  4037bd:	41 5c                	pop    %r12
  4037bf:	41 5d                	pop    %r13
  4037c1:	41 5e                	pop    %r14
  4037c3:	41 5f                	pop    %r15
  4037c5:	5d                   	pop    %rbp
  4037c6:	c3                   	retq   
  4037c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4037ce:	00 00 
  4037d0:	4c 89 e7             	mov    %r12,%rdi
  4037d3:	e8 a8 dd ff ff       	callq  401580 <_ZN2cv3Mat10deallocateEv@plt>
  4037d8:	e9 9b fe ff ff       	jmpq   403678 <_Z11decodeImagePfS_S_+0x5e8>
  4037dd:	0f 1f 00             	nopl   (%rax)
  4037e0:	89 44 24 64          	mov    %eax,0x64(%rsp)
  4037e4:	8b 84 24 28 01 00 00 	mov    0x128(%rsp),%eax
  4037eb:	89 44 24 68          	mov    %eax,0x68(%rsp)
  4037ef:	8b 84 24 2c 01 00 00 	mov    0x12c(%rsp),%eax
  4037f6:	89 44 24 6c          	mov    %eax,0x6c(%rsp)
  4037fa:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  403801:	00 
  403802:	48 8b 10             	mov    (%rax),%rdx
  403805:	48 8b 84 24 a8 00 00 	mov    0xa8(%rsp),%rax
  40380c:	00 
  40380d:	48 89 10             	mov    %rdx,(%rax)
  403810:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  403817:	00 
  403818:	48 8b 50 08          	mov    0x8(%rax),%rdx
  40381c:	48 8b 84 24 a8 00 00 	mov    0xa8(%rsp),%rax
  403823:	00 
  403824:	48 89 50 08          	mov    %rdx,0x8(%rax)
  403828:	e9 a9 fc ff ff       	jmpq   4034d6 <_Z11decodeImagePfS_S_+0x446>
  40382d:	0f 1f 00             	nopl   (%rax)
  403830:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  403835:	e8 46 dd ff ff       	callq  401580 <_ZN2cv3Mat10deallocateEv@plt>
  40383a:	e9 1f fc ff ff       	jmpq   40345e <_Z11decodeImagePfS_S_+0x3ce>
  40383f:	90                   	nop
  403840:	48 8d bc 24 20 01 00 	lea    0x120(%rsp),%rdi
  403847:	00 
  403848:	e8 33 dd ff ff       	callq  401580 <_ZN2cv3Mat10deallocateEv@plt>
  40384d:	e9 ce fc ff ff       	jmpq   403520 <_Z11decodeImagePfS_S_+0x490>
  403852:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  403858:	48 8d bc 24 c0 00 00 	lea    0xc0(%rsp),%rdi
  40385f:	00 
  403860:	e8 1b dd ff ff       	callq  401580 <_ZN2cv3Mat10deallocateEv@plt>
  403865:	e9 76 fe ff ff       	jmpq   4036e0 <_Z11decodeImagePfS_S_+0x650>
  40386a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  403870:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  403875:	e8 06 dd ff ff       	callq  401580 <_ZN2cv3Mat10deallocateEv@plt>
  40387a:	e9 d9 fe ff ff       	jmpq   403758 <_Z11decodeImagePfS_S_+0x6c8>
  40387f:	b9 a0 16 40 00       	mov    $0x4016a0,%ecx
  403884:	48 8d 57 10          	lea    0x10(%rdi),%rdx
  403888:	48 85 c9             	test   %rcx,%rcx
  40388b:	74 46                	je     4038d3 <_Z11decodeImagePfS_S_+0x843>
  40388d:	83 c8 ff             	or     $0xffffffff,%eax
  403890:	f0 0f c1 02          	lock xadd %eax,(%rdx)
  403894:	85 c0                	test   %eax,%eax
  403896:	0f 8f fe fc ff ff    	jg     40359a <_Z11decodeImagePfS_S_+0x50a>
  40389c:	48 8d b4 24 20 01 00 	lea    0x120(%rsp),%rsi
  4038a3:	00 
  4038a4:	e8 17 de ff ff       	callq  4016c0 <_ZNSs4_Rep10_M_destroyERKSaIcE@plt>
  4038a9:	e9 ec fc ff ff       	jmpq   40359a <_Z11decodeImagePfS_S_+0x50a>
  4038ae:	48 89 c3             	mov    %rax,%rbx
  4038b1:	c5 f8 77             	vzeroupper 
  4038b4:	48 8d bc 24 c0 00 00 	lea    0xc0(%rsp),%rdi
  4038bb:	00 
  4038bc:	e8 8f 00 00 00       	callq  403950 <_ZN2cv3MatD1Ev>
  4038c1:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  4038c6:	e8 85 00 00 00       	callq  403950 <_ZN2cv3MatD1Ev>
  4038cb:	48 89 df             	mov    %rbx,%rdi
  4038ce:	e8 0d dd ff ff       	callq  4015e0 <_Unwind_Resume@plt>
  4038d3:	8b 50 f8             	mov    -0x8(%rax),%edx
  4038d6:	8d 4a ff             	lea    -0x1(%rdx),%ecx
  4038d9:	89 48 f8             	mov    %ecx,-0x8(%rax)
  4038dc:	89 d0                	mov    %edx,%eax
  4038de:	eb b4                	jmp    403894 <_Z11decodeImagePfS_S_+0x804>
  4038e0:	49 89 c4             	mov    %rax,%r12
  4038e3:	c5 f8 77             	vzeroupper 
  4038e6:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  4038eb:	48 8d 74 24 1f       	lea    0x1f(%rsp),%rsi
  4038f0:	48 8d 78 e8          	lea    -0x18(%rax),%rdi
  4038f4:	e8 47 dd ff ff       	callq  401640 <_ZNSs4_Rep10_M_disposeERKSaIcE@plt>
  4038f9:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  4038fe:	48 85 ff             	test   %rdi,%rdi
  403901:	74 05                	je     403908 <_Z11decodeImagePfS_S_+0x878>
  403903:	e8 88 dc ff ff       	callq  401590 <_ZdlPv@plt>
  403908:	48 8d bb c0 00 00 00 	lea    0xc0(%rbx),%rdi
  40390f:	e8 3c 00 00 00       	callq  403950 <_ZN2cv3MatD1Ev>
  403914:	48 8d 7b 60          	lea    0x60(%rbx),%rdi
  403918:	e8 33 00 00 00       	callq  403950 <_ZN2cv3MatD1Ev>
  40391d:	48 89 df             	mov    %rbx,%rdi
  403920:	4c 89 e3             	mov    %r12,%rbx
  403923:	e8 28 00 00 00       	callq  403950 <_ZN2cv3MatD1Ev>
  403928:	eb 8a                	jmp    4038b4 <_Z11decodeImagePfS_S_+0x824>
  40392a:	49 89 c4             	mov    %rax,%r12
  40392d:	c5 f8 77             	vzeroupper 
  403930:	eb c7                	jmp    4038f9 <_Z11decodeImagePfS_S_+0x869>
  403932:	48 89 c3             	mov    %rax,%rbx
  403935:	c5 f8 77             	vzeroupper 
  403938:	eb 87                	jmp    4038c1 <_Z11decodeImagePfS_S_+0x831>
  40393a:	48 8d bc 24 20 01 00 	lea    0x120(%rsp),%rdi
  403941:	00 
  403942:	49 89 c4             	mov    %rax,%r12
  403945:	c5 f8 77             	vzeroupper 
  403948:	e8 03 00 00 00       	callq  403950 <_ZN2cv3MatD1Ev>
  40394d:	eb 97                	jmp    4038e6 <_Z11decodeImagePfS_S_+0x856>
  40394f:	90                   	nop

0000000000403950 <_ZN2cv3MatD1Ev>:
  403950:	53                   	push   %rbx
  403951:	48 8b 47 18          	mov    0x18(%rdi),%rax
  403955:	48 89 fb             	mov    %rdi,%rbx
  403958:	48 85 c0             	test   %rax,%rax
  40395b:	74 06                	je     403963 <_ZN2cv3MatD1Ev+0x13>
  40395d:	f0 83 28 01          	lock subl $0x1,(%rax)
  403961:	74 55                	je     4039b8 <_ZN2cv3MatD1Ev+0x68>
  403963:	48 8b 43 40          	mov    0x40(%rbx),%rax
  403967:	48 8b 7b 48          	mov    0x48(%rbx),%rdi
  40396b:	48 83 c3 50          	add    $0x50,%rbx
  40396f:	48 c7 43 e0 00 00 00 	movq   $0x0,-0x20(%rbx)
  403976:	00 
  403977:	48 c7 43 d8 00 00 00 	movq   $0x0,-0x28(%rbx)
  40397e:	00 
  40397f:	48 c7 43 d0 00 00 00 	movq   $0x0,-0x30(%rbx)
  403986:	00 
  403987:	48 c7 43 c0 00 00 00 	movq   $0x0,-0x40(%rbx)
  40398e:	00 
  40398f:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  403995:	48 c7 43 c8 00 00 00 	movq   $0x0,-0x38(%rbx)
  40399c:	00 
  40399d:	48 39 df             	cmp    %rbx,%rdi
  4039a0:	74 0e                	je     4039b0 <_ZN2cv3MatD1Ev+0x60>
  4039a2:	5b                   	pop    %rbx
  4039a3:	e9 e8 dc ff ff       	jmpq   401690 <_ZN2cv8fastFreeEPv@plt>
  4039a8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4039af:	00 
  4039b0:	5b                   	pop    %rbx
  4039b1:	c3                   	retq   
  4039b2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4039b8:	e8 c3 db ff ff       	callq  401580 <_ZN2cv3Mat10deallocateEv@plt>
  4039bd:	0f 1f 00             	nopl   (%rax)
  4039c0:	eb a1                	jmp    403963 <_ZN2cv3MatD1Ev+0x13>
  4039c2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4039c9:	00 00 00 
  4039cc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004039d0 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev>:
  4039d0:	41 54                	push   %r12
  4039d2:	49 89 fc             	mov    %rdi,%r12
  4039d5:	55                   	push   %rbp
  4039d6:	53                   	push   %rbx
  4039d7:	48 8b 6f 08          	mov    0x8(%rdi),%rbp
  4039db:	48 8b 1f             	mov    (%rdi),%rbx
  4039de:	48 39 dd             	cmp    %rbx,%rbp
  4039e1:	75 5a                	jne    403a3d <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x6d>
  4039e3:	e9 95 00 00 00       	jmpq   403a7d <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0xad>
  4039e8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4039ef:	00 
  4039f0:	48 8b 43 40          	mov    0x40(%rbx),%rax
  4039f4:	48 c7 43 30 00 00 00 	movq   $0x0,0x30(%rbx)
  4039fb:	00 
  4039fc:	48 c7 43 28 00 00 00 	movq   $0x0,0x28(%rbx)
  403a03:	00 
  403a04:	48 c7 43 20 00 00 00 	movq   $0x0,0x20(%rbx)
  403a0b:	00 
  403a0c:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
  403a13:	00 
  403a14:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  403a1a:	48 8b 7b 48          	mov    0x48(%rbx),%rdi
  403a1e:	48 8d 43 50          	lea    0x50(%rbx),%rax
  403a22:	48 c7 43 18 00 00 00 	movq   $0x0,0x18(%rbx)
  403a29:	00 
  403a2a:	48 39 c7             	cmp    %rax,%rdi
  403a2d:	74 05                	je     403a34 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x64>
  403a2f:	e8 5c dc ff ff       	callq  401690 <_ZN2cv8fastFreeEPv@plt>
  403a34:	48 83 c3 60          	add    $0x60,%rbx
  403a38:	48 39 dd             	cmp    %rbx,%rbp
  403a3b:	74 23                	je     403a60 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x90>
  403a3d:	48 8b 43 18          	mov    0x18(%rbx),%rax
  403a41:	48 85 c0             	test   %rax,%rax
  403a44:	74 aa                	je     4039f0 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x20>
  403a46:	f0 83 28 01          	lock subl $0x1,(%rax)
  403a4a:	75 a4                	jne    4039f0 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x20>
  403a4c:	48 89 df             	mov    %rbx,%rdi
  403a4f:	e8 2c db ff ff       	callq  401580 <_ZN2cv3Mat10deallocateEv@plt>
  403a54:	eb 9a                	jmp    4039f0 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x20>
  403a56:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  403a5d:	00 00 00 
  403a60:	49 8b 3c 24          	mov    (%r12),%rdi
  403a64:	48 85 ff             	test   %rdi,%rdi
  403a67:	74 0f                	je     403a78 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0xa8>
  403a69:	5b                   	pop    %rbx
  403a6a:	5d                   	pop    %rbp
  403a6b:	41 5c                	pop    %r12
  403a6d:	e9 1e db ff ff       	jmpq   401590 <_ZdlPv@plt>
  403a72:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  403a78:	5b                   	pop    %rbx
  403a79:	5d                   	pop    %rbp
  403a7a:	41 5c                	pop    %r12
  403a7c:	c3                   	retq   
  403a7d:	48 89 ef             	mov    %rbp,%rdi
  403a80:	eb e2                	jmp    403a64 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x94>
  403a82:	49 8b 3c 24          	mov    (%r12),%rdi
  403a86:	48 89 c3             	mov    %rax,%rbx
  403a89:	48 85 ff             	test   %rdi,%rdi
  403a8c:	74 10                	je     403a9e <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0xce>
  403a8e:	c5 f8 77             	vzeroupper 
  403a91:	e8 fa da ff ff       	callq  401590 <_ZdlPv@plt>
  403a96:	48 89 df             	mov    %rbx,%rdi
  403a99:	e8 42 db ff ff       	callq  4015e0 <_Unwind_Resume@plt>
  403a9e:	c5 f8 77             	vzeroupper 
  403aa1:	eb f3                	jmp    403a96 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0xc6>
  403aa3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  403aaa:	00 00 00 
  403aad:	0f 1f 00             	nopl   (%rax)

0000000000403ab0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_>:
  403ab0:	55                   	push   %rbp
  403ab1:	48 89 e5             	mov    %rsp,%rbp
  403ab4:	41 57                	push   %r15
  403ab6:	49 89 ff             	mov    %rdi,%r15
  403ab9:	41 56                	push   %r14
  403abb:	49 89 f6             	mov    %rsi,%r14
  403abe:	41 55                	push   %r13
  403ac0:	41 54                	push   %r12
  403ac2:	53                   	push   %rbx
  403ac3:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  403ac7:	48 81 ec 90 00 00 00 	sub    $0x90,%rsp
  403ace:	48 8b 7f 08          	mov    0x8(%rdi),%rdi
  403ad2:	49 3b 7f 10          	cmp    0x10(%r15),%rdi
  403ad6:	0f 84 3b 04 00 00    	je     403f17 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x467>
  403adc:	31 c0                	xor    %eax,%eax
  403ade:	48 85 ff             	test   %rdi,%rdi
  403ae1:	0f 84 8c 00 00 00    	je     403b73 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0xc3>
  403ae7:	48 8b 4f c0          	mov    -0x40(%rdi),%rcx
  403aeb:	48 8b 47 b0          	mov    -0x50(%rdi),%rax
  403aef:	c5 fa 6f 47 a0       	vmovdqu -0x60(%rdi),%xmm0
  403af4:	48 89 4f 20          	mov    %rcx,0x20(%rdi)
  403af8:	48 8b 4f c8          	mov    -0x38(%rdi),%rcx
  403afc:	48 89 47 10          	mov    %rax,0x10(%rdi)
  403b00:	48 8b 47 b8          	mov    -0x48(%rdi),%rax
  403b04:	c5 fa 7f 07          	vmovdqu %xmm0,(%rdi)
  403b08:	48 89 4f 28          	mov    %rcx,0x28(%rdi)
  403b0c:	48 8b 4f d0          	mov    -0x30(%rdi),%rcx
  403b10:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  403b14:	48 85 c0             	test   %rax,%rax
  403b17:	48 89 4f 30          	mov    %rcx,0x30(%rdi)
  403b1b:	48 8b 4f d8          	mov    -0x28(%rdi),%rcx
  403b1f:	48 89 47 18          	mov    %rax,0x18(%rdi)
  403b23:	c5 fa 7f 47 50       	vmovdqu %xmm0,0x50(%rdi)
  403b28:	48 89 4f 38          	mov    %rcx,0x38(%rdi)
  403b2c:	48 8d 4f 08          	lea    0x8(%rdi),%rcx
  403b30:	48 89 4f 40          	mov    %rcx,0x40(%rdi)
  403b34:	48 8d 4f 50          	lea    0x50(%rdi),%rcx
  403b38:	48 89 4f 48          	mov    %rcx,0x48(%rdi)
  403b3c:	8b 4f a4             	mov    -0x5c(%rdi),%ecx
  403b3f:	74 07                	je     403b48 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x98>
  403b41:	f0 83 00 01          	lock addl $0x1,(%rax)
  403b45:	8b 4f a4             	mov    -0x5c(%rdi),%ecx
  403b48:	83 f9 02             	cmp    $0x2,%ecx
  403b4b:	0f 8f 57 03 00 00    	jg     403ea8 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x3f8>
  403b51:	48 8b 4f e8          	mov    -0x18(%rdi),%rcx
  403b55:	48 8b 47 48          	mov    0x48(%rdi),%rax
  403b59:	48 8b 09             	mov    (%rcx),%rcx
  403b5c:	48 89 08             	mov    %rcx,(%rax)
  403b5f:	48 8b 4f e8          	mov    -0x18(%rdi),%rcx
  403b63:	48 8b 47 48          	mov    0x48(%rdi),%rax
  403b67:	48 8b 49 08          	mov    0x8(%rcx),%rcx
  403b6b:	48 89 48 08          	mov    %rcx,0x8(%rax)
  403b6f:	49 8b 47 08          	mov    0x8(%r15),%rax
  403b73:	48 8b 4a 20          	mov    0x20(%rdx),%rcx
  403b77:	48 83 c0 60          	add    $0x60,%rax
  403b7b:	4c 8d 6c 24 30       	lea    0x30(%rsp),%r13
  403b80:	49 89 47 08          	mov    %rax,0x8(%r15)
  403b84:	48 8b 42 10          	mov    0x10(%rdx),%rax
  403b88:	c5 fa 6f 02          	vmovdqu (%rdx),%xmm0
  403b8c:	48 89 4c 24 50       	mov    %rcx,0x50(%rsp)
  403b91:	48 8b 4a 28          	mov    0x28(%rdx),%rcx
  403b95:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  403b9a:	48 8b 42 18          	mov    0x18(%rdx),%rax
  403b9e:	c5 f9 7f 44 24 30    	vmovdqa %xmm0,0x30(%rsp)
  403ba4:	48 89 4c 24 58       	mov    %rcx,0x58(%rsp)
  403ba9:	48 8b 4a 30          	mov    0x30(%rdx),%rcx
  403bad:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  403bb1:	48 85 c0             	test   %rax,%rax
  403bb4:	48 89 4c 24 60       	mov    %rcx,0x60(%rsp)
  403bb9:	48 8b 4a 38          	mov    0x38(%rdx),%rcx
  403bbd:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  403bc2:	c5 f9 7f 84 24 80 00 	vmovdqa %xmm0,0x80(%rsp)
  403bc9:	00 00 
  403bcb:	48 89 4c 24 68       	mov    %rcx,0x68(%rsp)
  403bd0:	49 8d 4d 08          	lea    0x8(%r13),%rcx
  403bd4:	48 89 4c 24 70       	mov    %rcx,0x70(%rsp)
  403bd9:	49 8d 4d 50          	lea    0x50(%r13),%rcx
  403bdd:	48 89 4c 24 78       	mov    %rcx,0x78(%rsp)
  403be2:	8b 4a 04             	mov    0x4(%rdx),%ecx
  403be5:	74 07                	je     403bee <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x13e>
  403be7:	f0 83 00 01          	lock addl $0x1,(%rax)
  403beb:	8b 4a 04             	mov    0x4(%rdx),%ecx
  403bee:	83 f9 02             	cmp    $0x2,%ecx
  403bf1:	0f 8f d9 02 00 00    	jg     403ed0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x420>
  403bf7:	48 8b 42 48          	mov    0x48(%rdx),%rax
  403bfb:	48 8b 08             	mov    (%rax),%rcx
  403bfe:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  403c03:	48 89 08             	mov    %rcx,(%rax)
  403c06:	48 8b 42 48          	mov    0x48(%rdx),%rax
  403c0a:	48 8b 50 08          	mov    0x8(%rax),%rdx
  403c0e:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  403c13:	48 89 50 08          	mov    %rdx,0x8(%rax)
  403c17:	4d 8b 67 08          	mov    0x8(%r15),%r12
  403c1b:	48 b8 ab aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaab,%rax
  403c22:	aa aa aa 
  403c25:	49 8d 5c 24 a0       	lea    -0x60(%r12),%rbx
  403c2a:	49 81 ec c0 00 00 00 	sub    $0xc0,%r12
  403c31:	4d 29 f4             	sub    %r14,%r12
  403c34:	49 c1 fc 05          	sar    $0x5,%r12
  403c38:	4c 0f af e0          	imul   %rax,%r12
  403c3c:	4d 85 e4             	test   %r12,%r12
  403c3f:	7f 76                	jg     403cb7 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x207>
  403c41:	e9 0a 01 00 00       	jmpq   403d50 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2a0>
  403c46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  403c4d:	00 00 00 
  403c50:	89 43 04             	mov    %eax,0x4(%rbx)
  403c53:	8b 43 a8             	mov    -0x58(%rbx),%eax
  403c56:	48 8b 53 e8          	mov    -0x18(%rbx),%rdx
  403c5a:	89 43 08             	mov    %eax,0x8(%rbx)
  403c5d:	8b 43 ac             	mov    -0x54(%rbx),%eax
  403c60:	89 43 0c             	mov    %eax,0xc(%rbx)
  403c63:	48 8b 43 48          	mov    0x48(%rbx),%rax
  403c67:	48 8b 12             	mov    (%rdx),%rdx
  403c6a:	48 89 10             	mov    %rdx,(%rax)
  403c6d:	48 8b 53 e8          	mov    -0x18(%rbx),%rdx
  403c71:	48 8b 43 48          	mov    0x48(%rbx),%rax
  403c75:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  403c79:	48 89 50 08          	mov    %rdx,0x8(%rax)
  403c7d:	48 8b 43 b0          	mov    -0x50(%rbx),%rax
  403c81:	48 89 43 10          	mov    %rax,0x10(%rbx)
  403c85:	48 8b 43 c0          	mov    -0x40(%rbx),%rax
  403c89:	48 89 43 20          	mov    %rax,0x20(%rbx)
  403c8d:	48 8b 43 c8          	mov    -0x38(%rbx),%rax
  403c91:	48 89 43 28          	mov    %rax,0x28(%rbx)
  403c95:	48 8b 43 d0          	mov    -0x30(%rbx),%rax
  403c99:	48 89 43 30          	mov    %rax,0x30(%rbx)
  403c9d:	48 8b 43 b8          	mov    -0x48(%rbx),%rax
  403ca1:	48 89 43 18          	mov    %rax,0x18(%rbx)
  403ca5:	48 8b 43 d8          	mov    -0x28(%rbx),%rax
  403ca9:	48 89 43 38          	mov    %rax,0x38(%rbx)
  403cad:	49 83 ec 01          	sub    $0x1,%r12
  403cb1:	0f 84 99 00 00 00    	je     403d50 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2a0>
  403cb7:	4c 8d bb 40 ff ff ff 	lea    -0xc0(%rbx),%r15
  403cbe:	48 83 eb 60          	sub    $0x60,%rbx
  403cc2:	4c 39 fb             	cmp    %r15,%rbx
  403cc5:	74 e6                	je     403cad <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x1fd>
  403cc7:	48 8b 43 b8          	mov    -0x48(%rbx),%rax
  403ccb:	48 85 c0             	test   %rax,%rax
  403cce:	74 04                	je     403cd4 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x224>
  403cd0:	f0 83 00 01          	lock addl $0x1,(%rax)
  403cd4:	48 8b 43 18          	mov    0x18(%rbx),%rax
  403cd8:	48 85 c0             	test   %rax,%rax
  403cdb:	74 06                	je     403ce3 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x233>
  403cdd:	f0 83 28 01          	lock subl $0x1,(%rax)
  403ce1:	74 5d                	je     403d40 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x290>
  403ce3:	48 8b 43 40          	mov    0x40(%rbx),%rax
  403ce7:	48 c7 43 30 00 00 00 	movq   $0x0,0x30(%rbx)
  403cee:	00 
  403cef:	48 c7 43 28 00 00 00 	movq   $0x0,0x28(%rbx)
  403cf6:	00 
  403cf7:	48 c7 43 20 00 00 00 	movq   $0x0,0x20(%rbx)
  403cfe:	00 
  403cff:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
  403d06:	00 
  403d07:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  403d0d:	83 7b 04 02          	cmpl   $0x2,0x4(%rbx)
  403d11:	8b 43 a0             	mov    -0x60(%rbx),%eax
  403d14:	48 c7 43 18 00 00 00 	movq   $0x0,0x18(%rbx)
  403d1b:	00 
  403d1c:	89 03                	mov    %eax,(%rbx)
  403d1e:	7f 0c                	jg     403d2c <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x27c>
  403d20:	8b 43 a4             	mov    -0x5c(%rbx),%eax
  403d23:	83 f8 02             	cmp    $0x2,%eax
  403d26:	0f 8e 24 ff ff ff    	jle    403c50 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x1a0>
  403d2c:	4c 89 fe             	mov    %r15,%rsi
  403d2f:	48 89 df             	mov    %rbx,%rdi
  403d32:	e8 89 d7 ff ff       	callq  4014c0 <_ZN2cv3Mat8copySizeERKS0_@plt>
  403d37:	e9 41 ff ff ff       	jmpq   403c7d <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x1cd>
  403d3c:	0f 1f 40 00          	nopl   0x0(%rax)
  403d40:	48 89 df             	mov    %rbx,%rdi
  403d43:	e8 38 d8 ff ff       	callq  401580 <_ZN2cv3Mat10deallocateEv@plt>
  403d48:	eb 99                	jmp    403ce3 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x233>
  403d4a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  403d50:	4d 39 ee             	cmp    %r13,%r14
  403d53:	0f 84 b4 01 00 00    	je     403f0d <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x45d>
  403d59:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  403d5e:	48 85 c0             	test   %rax,%rax
  403d61:	74 04                	je     403d67 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2b7>
  403d63:	f0 83 00 01          	lock addl $0x1,(%rax)
  403d67:	49 8b 46 18          	mov    0x18(%r14),%rax
  403d6b:	48 85 c0             	test   %rax,%rax
  403d6e:	74 0a                	je     403d7a <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2ca>
  403d70:	f0 83 28 01          	lock subl $0x1,(%rax)
  403d74:	0f 84 86 01 00 00    	je     403f00 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x450>
  403d7a:	49 8b 46 40          	mov    0x40(%r14),%rax
  403d7e:	49 c7 46 30 00 00 00 	movq   $0x0,0x30(%r14)
  403d85:	00 
  403d86:	49 c7 46 28 00 00 00 	movq   $0x0,0x28(%r14)
  403d8d:	00 
  403d8e:	49 c7 46 20 00 00 00 	movq   $0x0,0x20(%r14)
  403d95:	00 
  403d96:	49 c7 46 10 00 00 00 	movq   $0x0,0x10(%r14)
  403d9d:	00 
  403d9e:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  403da4:	41 83 7e 04 02       	cmpl   $0x2,0x4(%r14)
  403da9:	8b 44 24 30          	mov    0x30(%rsp),%eax
  403dad:	49 c7 46 18 00 00 00 	movq   $0x0,0x18(%r14)
  403db4:	00 
  403db5:	41 89 06             	mov    %eax,(%r14)
  403db8:	7f 0d                	jg     403dc7 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x317>
  403dba:	8b 44 24 34          	mov    0x34(%rsp),%eax
  403dbe:	83 f8 02             	cmp    $0x2,%eax
  403dc1:	0f 8e a1 00 00 00    	jle    403e68 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x3b8>
  403dc7:	4c 89 ee             	mov    %r13,%rsi
  403dca:	4c 89 f7             	mov    %r14,%rdi
  403dcd:	e8 ee d6 ff ff       	callq  4014c0 <_ZN2cv3Mat8copySizeERKS0_@plt>
  403dd2:	c5 f9 6f 44 24 40    	vmovdqa 0x40(%rsp),%xmm0
  403dd8:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  403ddd:	c4 c1 7a 7f 46 10    	vmovdqu %xmm0,0x10(%r14)
  403de3:	c5 f9 6f 44 24 50    	vmovdqa 0x50(%rsp),%xmm0
  403de9:	c4 c1 7a 7f 46 20    	vmovdqu %xmm0,0x20(%r14)
  403def:	c5 f9 6f 44 24 60    	vmovdqa 0x60(%rsp),%xmm0
  403df5:	c4 c1 7a 7f 46 30    	vmovdqu %xmm0,0x30(%r14)
  403dfb:	48 85 c0             	test   %rax,%rax
  403dfe:	74 0a                	je     403e0a <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x35a>
  403e00:	f0 83 28 01          	lock subl $0x1,(%rax)
  403e04:	0f 84 e6 00 00 00    	je     403ef0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x440>
  403e0a:	48 8b 7c 24 78       	mov    0x78(%rsp),%rdi
  403e0f:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
  403e14:	49 83 c5 50          	add    $0x50,%r13
  403e18:	48 c7 44 24 60 00 00 	movq   $0x0,0x60(%rsp)
  403e1f:	00 00 
  403e21:	48 c7 44 24 58 00 00 	movq   $0x0,0x58(%rsp)
  403e28:	00 00 
  403e2a:	48 c7 44 24 50 00 00 	movq   $0x0,0x50(%rsp)
  403e31:	00 00 
  403e33:	48 c7 44 24 40 00 00 	movq   $0x0,0x40(%rsp)
  403e3a:	00 00 
  403e3c:	4c 39 ef             	cmp    %r13,%rdi
  403e3f:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  403e45:	48 c7 44 24 48 00 00 	movq   $0x0,0x48(%rsp)
  403e4c:	00 00 
  403e4e:	74 05                	je     403e55 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x3a5>
  403e50:	e8 3b d8 ff ff       	callq  401690 <_ZN2cv8fastFreeEPv@plt>
  403e55:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
  403e59:	5b                   	pop    %rbx
  403e5a:	41 5c                	pop    %r12
  403e5c:	41 5d                	pop    %r13
  403e5e:	41 5e                	pop    %r14
  403e60:	41 5f                	pop    %r15
  403e62:	5d                   	pop    %rbp
  403e63:	c3                   	retq   
  403e64:	0f 1f 40 00          	nopl   0x0(%rax)
  403e68:	41 89 46 04          	mov    %eax,0x4(%r14)
  403e6c:	8b 44 24 38          	mov    0x38(%rsp),%eax
  403e70:	48 8b 54 24 78       	mov    0x78(%rsp),%rdx
  403e75:	41 89 46 08          	mov    %eax,0x8(%r14)
  403e79:	8b 44 24 3c          	mov    0x3c(%rsp),%eax
  403e7d:	48 8b 12             	mov    (%rdx),%rdx
  403e80:	41 89 46 0c          	mov    %eax,0xc(%r14)
  403e84:	49 8b 46 48          	mov    0x48(%r14),%rax
  403e88:	48 89 10             	mov    %rdx,(%rax)
  403e8b:	48 8b 54 24 78       	mov    0x78(%rsp),%rdx
  403e90:	49 8b 46 48          	mov    0x48(%r14),%rax
  403e94:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  403e98:	48 89 50 08          	mov    %rdx,0x8(%rax)
  403e9c:	e9 31 ff ff ff       	jmpq   403dd2 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x322>
  403ea1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  403ea8:	48 8d 77 a0          	lea    -0x60(%rdi),%rsi
  403eac:	c7 47 04 00 00 00 00 	movl   $0x0,0x4(%rdi)
  403eb3:	48 89 54 24 28       	mov    %rdx,0x28(%rsp)
  403eb8:	e8 03 d6 ff ff       	callq  4014c0 <_ZN2cv3Mat8copySizeERKS0_@plt>
  403ebd:	49 8b 47 08          	mov    0x8(%r15),%rax
  403ec1:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
  403ec6:	e9 a8 fc ff ff       	jmpq   403b73 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0xc3>
  403ecb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  403ed0:	48 89 d6             	mov    %rdx,%rsi
  403ed3:	4c 89 ef             	mov    %r13,%rdi
  403ed6:	c7 44 24 34 00 00 00 	movl   $0x0,0x34(%rsp)
  403edd:	00 
  403ede:	e8 dd d5 ff ff       	callq  4014c0 <_ZN2cv3Mat8copySizeERKS0_@plt>
  403ee3:	e9 2f fd ff ff       	jmpq   403c17 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x167>
  403ee8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  403eef:	00 
  403ef0:	4c 89 ef             	mov    %r13,%rdi
  403ef3:	e8 88 d6 ff ff       	callq  401580 <_ZN2cv3Mat10deallocateEv@plt>
  403ef8:	e9 0d ff ff ff       	jmpq   403e0a <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x35a>
  403efd:	0f 1f 00             	nopl   (%rax)
  403f00:	4c 89 f7             	mov    %r14,%rdi
  403f03:	e8 78 d6 ff ff       	callq  401580 <_ZN2cv3Mat10deallocateEv@plt>
  403f08:	e9 6d fe ff ff       	jmpq   403d7a <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2ca>
  403f0d:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  403f12:	e9 e4 fe ff ff       	jmpq   403dfb <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x34b>
  403f17:	49 8b 07             	mov    (%r15),%rax
  403f1a:	48 b9 ab aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaab,%rcx
  403f21:	aa aa aa 
  403f24:	48 29 c7             	sub    %rax,%rdi
  403f27:	48 c1 ff 05          	sar    $0x5,%rdi
  403f2b:	48 0f af f9          	imul   %rcx,%rdi
  403f2f:	48 85 ff             	test   %rdi,%rdi
  403f32:	0f 85 be 01 00 00    	jne    4040f6 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x646>
  403f38:	48 89 f3             	mov    %rsi,%rbx
  403f3b:	be 01 00 00 00       	mov    $0x1,%esi
  403f40:	48 29 c3             	sub    %rax,%rbx
  403f43:	48 c1 fb 05          	sar    $0x5,%rbx
  403f47:	48 0f af d9          	imul   %rcx,%rbx
  403f4b:	48 8d 3c 76          	lea    (%rsi,%rsi,2),%rdi
  403f4f:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  403f54:	48 c1 e7 05          	shl    $0x5,%rdi
  403f58:	48 89 54 24 28       	mov    %rdx,0x28(%rsp)
  403f5d:	e8 7e d5 ff ff       	callq  4014e0 <_Znwm@plt>
  403f62:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
  403f67:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  403f6c:	48 8d 04 5b          	lea    (%rbx,%rbx,2),%rax
  403f70:	48 c1 e0 05          	shl    $0x5,%rax
  403f74:	48 03 44 24 18       	add    0x18(%rsp),%rax
  403f79:	48 89 c6             	mov    %rax,%rsi
  403f7c:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  403f81:	0f 84 87 00 00 00    	je     40400e <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x55e>
  403f87:	c5 fa 6f 02          	vmovdqu (%rdx),%xmm0
  403f8b:	c5 fa 7f 00          	vmovdqu %xmm0,(%rax)
  403f8f:	48 8b 42 10          	mov    0x10(%rdx),%rax
  403f93:	48 8b 4a 20          	mov    0x20(%rdx),%rcx
  403f97:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  403f9b:	48 89 46 10          	mov    %rax,0x10(%rsi)
  403f9f:	48 8b 42 18          	mov    0x18(%rdx),%rax
  403fa3:	48 89 4e 20          	mov    %rcx,0x20(%rsi)
  403fa7:	48 8b 4a 28          	mov    0x28(%rdx),%rcx
  403fab:	48 85 c0             	test   %rax,%rax
  403fae:	48 89 46 18          	mov    %rax,0x18(%rsi)
  403fb2:	48 89 4e 28          	mov    %rcx,0x28(%rsi)
  403fb6:	48 8b 4a 30          	mov    0x30(%rdx),%rcx
  403fba:	48 89 4e 30          	mov    %rcx,0x30(%rsi)
  403fbe:	48 8b 4a 38          	mov    0x38(%rdx),%rcx
  403fc2:	c5 fa 7f 46 50       	vmovdqu %xmm0,0x50(%rsi)
  403fc7:	48 89 4e 38          	mov    %rcx,0x38(%rsi)
  403fcb:	48 8d 4e 08          	lea    0x8(%rsi),%rcx
  403fcf:	48 89 4e 40          	mov    %rcx,0x40(%rsi)
  403fd3:	48 8d 4e 50          	lea    0x50(%rsi),%rcx
  403fd7:	48 89 4e 48          	mov    %rcx,0x48(%rsi)
  403fdb:	74 04                	je     403fe1 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x531>
  403fdd:	f0 83 00 01          	lock addl $0x1,(%rax)
  403fe1:	83 7a 04 02          	cmpl   $0x2,0x4(%rdx)
  403fe5:	0f 8f 0f 03 00 00    	jg     4042fa <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x84a>
  403feb:	48 8b 4a 48          	mov    0x48(%rdx),%rcx
  403fef:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  403ff4:	48 8b 09             	mov    (%rcx),%rcx
  403ff7:	48 8b 46 48          	mov    0x48(%rsi),%rax
  403ffb:	48 89 08             	mov    %rcx,(%rax)
  403ffe:	48 8b 52 48          	mov    0x48(%rdx),%rdx
  404002:	48 8b 46 48          	mov    0x48(%rsi),%rax
  404006:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  40400a:	48 89 50 08          	mov    %rdx,0x8(%rax)
  40400e:	4d 8b 27             	mov    (%r15),%r12
  404011:	48 8b 5c 24 18       	mov    0x18(%rsp),%rbx
  404016:	4d 39 e6             	cmp    %r12,%r14
  404019:	75 36                	jne    404051 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x5a1>
  40401b:	e9 16 01 00 00       	jmpq   404136 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x686>
  404020:	49 8b 54 24 48       	mov    0x48(%r12),%rdx
  404025:	48 8b 43 48          	mov    0x48(%rbx),%rax
  404029:	48 8b 12             	mov    (%rdx),%rdx
  40402c:	48 89 10             	mov    %rdx,(%rax)
  40402f:	49 8b 54 24 48       	mov    0x48(%r12),%rdx
  404034:	48 8b 43 48          	mov    0x48(%rbx),%rax
  404038:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  40403c:	48 89 50 08          	mov    %rdx,0x8(%rax)
  404040:	49 83 c4 60          	add    $0x60,%r12
  404044:	48 83 c3 60          	add    $0x60,%rbx
  404048:	4d 39 e6             	cmp    %r12,%r14
  40404b:	0f 84 e5 00 00 00    	je     404136 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x686>
  404051:	48 85 db             	test   %rbx,%rbx
  404054:	74 ea                	je     404040 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x590>
  404056:	41 8b 04 24          	mov    (%r12),%eax
  40405a:	89 03                	mov    %eax,(%rbx)
  40405c:	41 8b 44 24 04       	mov    0x4(%r12),%eax
  404061:	89 43 04             	mov    %eax,0x4(%rbx)
  404064:	41 8b 44 24 08       	mov    0x8(%r12),%eax
  404069:	89 43 08             	mov    %eax,0x8(%rbx)
  40406c:	41 8b 44 24 0c       	mov    0xc(%r12),%eax
  404071:	89 43 0c             	mov    %eax,0xc(%rbx)
  404074:	49 8b 44 24 10       	mov    0x10(%r12),%rax
  404079:	48 89 43 10          	mov    %rax,0x10(%rbx)
  40407d:	49 8b 44 24 18       	mov    0x18(%r12),%rax
  404082:	48 89 43 18          	mov    %rax,0x18(%rbx)
  404086:	49 8b 54 24 20       	mov    0x20(%r12),%rdx
  40408b:	48 85 c0             	test   %rax,%rax
  40408e:	48 89 53 20          	mov    %rdx,0x20(%rbx)
  404092:	49 8b 54 24 28       	mov    0x28(%r12),%rdx
  404097:	48 89 53 28          	mov    %rdx,0x28(%rbx)
  40409b:	49 8b 54 24 30       	mov    0x30(%r12),%rdx
  4040a0:	48 89 53 30          	mov    %rdx,0x30(%rbx)
  4040a4:	49 8b 54 24 38       	mov    0x38(%r12),%rdx
  4040a9:	48 c7 43 58 00 00 00 	movq   $0x0,0x58(%rbx)
  4040b0:	00 
  4040b1:	48 c7 43 50 00 00 00 	movq   $0x0,0x50(%rbx)
  4040b8:	00 
  4040b9:	48 89 53 38          	mov    %rdx,0x38(%rbx)
  4040bd:	48 8d 53 08          	lea    0x8(%rbx),%rdx
  4040c1:	48 89 53 40          	mov    %rdx,0x40(%rbx)
  4040c5:	48 8d 53 50          	lea    0x50(%rbx),%rdx
  4040c9:	48 89 53 48          	mov    %rdx,0x48(%rbx)
  4040cd:	74 04                	je     4040d3 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x623>
  4040cf:	f0 83 00 01          	lock addl $0x1,(%rax)
  4040d3:	41 83 7c 24 04 02    	cmpl   $0x2,0x4(%r12)
  4040d9:	0f 8e 41 ff ff ff    	jle    404020 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x570>
  4040df:	c7 43 04 00 00 00 00 	movl   $0x0,0x4(%rbx)
  4040e6:	4c 89 e6             	mov    %r12,%rsi
  4040e9:	48 89 df             	mov    %rbx,%rdi
  4040ec:	e8 cf d3 ff ff       	callq  4014c0 <_ZN2cv3Mat8copySizeERKS0_@plt>
  4040f1:	e9 4a ff ff ff       	jmpq   404040 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x590>
  4040f6:	48 8d 34 3f          	lea    (%rdi,%rdi,1),%rsi
  4040fa:	4c 89 f3             	mov    %r14,%rbx
  4040fd:	48 29 c3             	sub    %rax,%rbx
  404100:	48 c1 fb 05          	sar    $0x5,%rbx
  404104:	48 39 f7             	cmp    %rsi,%rdi
  404107:	0f 86 79 02 00 00    	jbe    404386 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x8d6>
  40410d:	48 b8 ab aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaab,%rax
  404114:	aa aa aa 
  404117:	48 c7 c7 c0 ff ff ff 	mov    $0xffffffffffffffc0,%rdi
  40411e:	48 0f af d8          	imul   %rax,%rbx
  404122:	48 b8 aa aa aa aa aa 	movabs $0x2aaaaaaaaaaaaaa,%rax
  404129:	aa aa 02 
  40412c:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  404131:	e9 22 fe ff ff       	jmpq   403f58 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x4a8>
  404136:	49 8b 77 08          	mov    0x8(%r15),%rsi
  40413a:	48 8d 43 60          	lea    0x60(%rbx),%rax
  40413e:	4d 89 f5             	mov    %r14,%r13
  404141:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  404146:	49 89 c4             	mov    %rax,%r12
  404149:	49 39 f6             	cmp    %rsi,%r14
  40414c:	48 89 74 24 20       	mov    %rsi,0x20(%rsp)
  404151:	75 40                	jne    404193 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x6e3>
  404153:	e9 fb 00 00 00       	jmpq   404253 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7a3>
  404158:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40415f:	00 
  404160:	49 8b 55 48          	mov    0x48(%r13),%rdx
  404164:	49 8b 44 24 48       	mov    0x48(%r12),%rax
  404169:	48 8b 12             	mov    (%rdx),%rdx
  40416c:	48 89 10             	mov    %rdx,(%rax)
  40416f:	49 8b 55 48          	mov    0x48(%r13),%rdx
  404173:	49 8b 44 24 48       	mov    0x48(%r12),%rax
  404178:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  40417c:	48 89 50 08          	mov    %rdx,0x8(%rax)
  404180:	49 83 c4 60          	add    $0x60,%r12
  404184:	49 83 c5 60          	add    $0x60,%r13
  404188:	4c 39 6c 24 20       	cmp    %r13,0x20(%rsp)
  40418d:	0f 84 bc 00 00 00    	je     40424f <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x79f>
  404193:	4d 85 e4             	test   %r12,%r12
  404196:	74 e8                	je     404180 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x6d0>
  404198:	41 8b 45 00          	mov    0x0(%r13),%eax
  40419c:	48 8b 74 24 28       	mov    0x28(%rsp),%rsi
  4041a1:	49 8d 4c 24 08       	lea    0x8(%r12),%rcx
  4041a6:	41 89 04 24          	mov    %eax,(%r12)
  4041aa:	41 8b 45 04          	mov    0x4(%r13),%eax
  4041ae:	41 89 44 24 04       	mov    %eax,0x4(%r12)
  4041b3:	41 8b 45 08          	mov    0x8(%r13),%eax
  4041b7:	41 89 44 24 08       	mov    %eax,0x8(%r12)
  4041bc:	41 8b 45 0c          	mov    0xc(%r13),%eax
  4041c0:	41 89 44 24 0c       	mov    %eax,0xc(%r12)
  4041c5:	49 8b 45 10          	mov    0x10(%r13),%rax
  4041c9:	49 89 44 24 10       	mov    %rax,0x10(%r12)
  4041ce:	49 8b 45 18          	mov    0x18(%r13),%rax
  4041d2:	49 89 44 24 18       	mov    %rax,0x18(%r12)
  4041d7:	49 8b 55 20          	mov    0x20(%r13),%rdx
  4041db:	49 89 54 24 20       	mov    %rdx,0x20(%r12)
  4041e0:	49 8b 55 28          	mov    0x28(%r13),%rdx
  4041e4:	49 89 54 24 28       	mov    %rdx,0x28(%r12)
  4041e9:	49 8b 55 30          	mov    0x30(%r13),%rdx
  4041ed:	49 89 54 24 30       	mov    %rdx,0x30(%r12)
  4041f2:	49 8b 55 38          	mov    0x38(%r13),%rdx
  4041f6:	49 89 54 24 38       	mov    %rdx,0x38(%r12)
  4041fb:	4c 89 e2             	mov    %r12,%rdx
  4041fe:	48 29 da             	sub    %rbx,%rdx
  404201:	48 85 c0             	test   %rax,%rax
  404204:	48 89 4c 16 e0       	mov    %rcx,-0x20(%rsi,%rdx,1)
  404209:	49 8d 4c 24 50       	lea    0x50(%r12),%rcx
  40420e:	48 89 4c 16 e8       	mov    %rcx,-0x18(%rsi,%rdx,1)
  404213:	49 c7 44 24 58 00 00 	movq   $0x0,0x58(%r12)
  40421a:	00 00 
  40421c:	49 c7 44 24 50 00 00 	movq   $0x0,0x50(%r12)
  404223:	00 00 
  404225:	74 04                	je     40422b <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x77b>
  404227:	f0 83 00 01          	lock addl $0x1,(%rax)
  40422b:	41 83 7d 04 02       	cmpl   $0x2,0x4(%r13)
  404230:	0f 8e 2a ff ff ff    	jle    404160 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x6b0>
  404236:	41 c7 44 24 04 00 00 	movl   $0x0,0x4(%r12)
  40423d:	00 00 
  40423f:	4c 89 ee             	mov    %r13,%rsi
  404242:	4c 89 e7             	mov    %r12,%rdi
  404245:	e8 76 d2 ff ff       	callq  4014c0 <_ZN2cv3Mat8copySizeERKS0_@plt>
  40424a:	e9 31 ff ff ff       	jmpq   404180 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x6d0>
  40424f:	4d 8b 77 08          	mov    0x8(%r15),%r14
  404253:	49 8b 1f             	mov    (%r15),%rbx
  404256:	4c 89 f7             	mov    %r14,%rdi
  404259:	4c 39 f3             	cmp    %r14,%rbx
  40425c:	75 4f                	jne    4042ad <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7fd>
  40425e:	eb 69                	jmp    4042c9 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x819>
  404260:	48 8b 43 40          	mov    0x40(%rbx),%rax
  404264:	48 c7 43 30 00 00 00 	movq   $0x0,0x30(%rbx)
  40426b:	00 
  40426c:	48 c7 43 28 00 00 00 	movq   $0x0,0x28(%rbx)
  404273:	00 
  404274:	48 c7 43 20 00 00 00 	movq   $0x0,0x20(%rbx)
  40427b:	00 
  40427c:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
  404283:	00 
  404284:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40428a:	48 8b 7b 48          	mov    0x48(%rbx),%rdi
  40428e:	48 8d 43 50          	lea    0x50(%rbx),%rax
  404292:	48 c7 43 18 00 00 00 	movq   $0x0,0x18(%rbx)
  404299:	00 
  40429a:	48 39 c7             	cmp    %rax,%rdi
  40429d:	74 05                	je     4042a4 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7f4>
  40429f:	e8 ec d3 ff ff       	callq  401690 <_ZN2cv8fastFreeEPv@plt>
  4042a4:	48 83 c3 60          	add    $0x60,%rbx
  4042a8:	4c 39 f3             	cmp    %r14,%rbx
  4042ab:	74 19                	je     4042c6 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x816>
  4042ad:	48 8b 43 18          	mov    0x18(%rbx),%rax
  4042b1:	48 85 c0             	test   %rax,%rax
  4042b4:	74 aa                	je     404260 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7b0>
  4042b6:	f0 83 28 01          	lock subl $0x1,(%rax)
  4042ba:	75 a4                	jne    404260 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7b0>
  4042bc:	48 89 df             	mov    %rbx,%rdi
  4042bf:	e8 bc d2 ff ff       	callq  401580 <_ZN2cv3Mat10deallocateEv@plt>
  4042c4:	eb 9a                	jmp    404260 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7b0>
  4042c6:	49 8b 3f             	mov    (%r15),%rdi
  4042c9:	48 85 ff             	test   %rdi,%rdi
  4042cc:	74 07                	je     4042d5 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x825>
  4042ce:	66 90                	xchg   %ax,%ax
  4042d0:	e8 bb d2 ff ff       	callq  401590 <_ZdlPv@plt>
  4042d5:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  4042da:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
  4042df:	4d 89 67 08          	mov    %r12,0x8(%r15)
  4042e3:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  4042e7:	49 89 37             	mov    %rsi,(%r15)
  4042ea:	48 c1 e0 05          	shl    $0x5,%rax
  4042ee:	48 01 f0             	add    %rsi,%rax
  4042f1:	49 89 47 10          	mov    %rax,0x10(%r15)
  4042f5:	e9 5b fb ff ff       	jmpq   403e55 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x3a5>
  4042fa:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4042ff:	48 89 d6             	mov    %rdx,%rsi
  404302:	c7 40 04 00 00 00 00 	movl   $0x0,0x4(%rax)
  404309:	48 89 c7             	mov    %rax,%rdi
  40430c:	e8 af d1 ff ff       	callq  4014c0 <_ZN2cv3Mat8copySizeERKS0_@plt>
  404311:	e9 f8 fc ff ff       	jmpq   40400e <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x55e>
  404316:	48 89 c7             	mov    %rax,%rdi
  404319:	c5 f8 77             	vzeroupper 
  40431c:	e8 7f d1 ff ff       	callq  4014a0 <__cxa_begin_catch@plt>
  404321:	4c 39 64 24 28       	cmp    %r12,0x28(%rsp)
  404326:	74 16                	je     40433e <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x88e>
  404328:	48 8b 5c 24 28       	mov    0x28(%rsp),%rbx
  40432d:	48 89 df             	mov    %rbx,%rdi
  404330:	e8 1b f6 ff ff       	callq  403950 <_ZN2cv3MatD1Ev>
  404335:	48 83 c3 60          	add    $0x60,%rbx
  404339:	4c 39 e3             	cmp    %r12,%rbx
  40433c:	75 ef                	jne    40432d <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x87d>
  40433e:	e8 0d d3 ff ff       	callq  401650 <__cxa_rethrow@plt>
  404343:	48 89 c3             	mov    %rax,%rbx
  404346:	4c 89 ef             	mov    %r13,%rdi
  404349:	c5 f8 77             	vzeroupper 
  40434c:	e8 ff f5 ff ff       	callq  403950 <_ZN2cv3MatD1Ev>
  404351:	48 89 df             	mov    %rbx,%rdi
  404354:	e8 87 d2 ff ff       	callq  4015e0 <_Unwind_Resume@plt>
  404359:	48 89 c7             	mov    %rax,%rdi
  40435c:	c5 f8 77             	vzeroupper 
  40435f:	e8 3c d1 ff ff       	callq  4014a0 <__cxa_begin_catch@plt>
  404364:	48 39 5c 24 18       	cmp    %rbx,0x18(%rsp)
  404369:	74 16                	je     404381 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x8d1>
  40436b:	4c 8b 64 24 18       	mov    0x18(%rsp),%r12
  404370:	4c 89 e7             	mov    %r12,%rdi
  404373:	e8 d8 f5 ff ff       	callq  403950 <_ZN2cv3MatD1Ev>
  404378:	49 83 c4 60          	add    $0x60,%r12
  40437c:	4c 39 e3             	cmp    %r12,%rbx
  40437f:	75 ef                	jne    404370 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x8c0>
  404381:	e8 ca d2 ff ff       	callq  401650 <__cxa_rethrow@plt>
  404386:	48 bf aa aa aa aa aa 	movabs $0x2aaaaaaaaaaaaaa,%rdi
  40438d:	aa aa 02 
  404390:	48 39 fe             	cmp    %rdi,%rsi
  404393:	0f 87 74 fd ff ff    	ja     40410d <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x65d>
  404399:	48 0f af d9          	imul   %rcx,%rbx
  40439d:	48 85 f6             	test   %rsi,%rsi
  4043a0:	0f 85 a5 fb ff ff    	jne    403f4b <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x49b>
  4043a6:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
  4043ad:	00 00 
  4043af:	48 c7 44 24 18 00 00 	movq   $0x0,0x18(%rsp)
  4043b6:	00 00 
  4043b8:	e9 af fb ff ff       	jmpq   403f6c <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x4bc>
  4043bd:	48 89 c7             	mov    %rax,%rdi
  4043c0:	c5 f8 77             	vzeroupper 
  4043c3:	e8 d8 d0 ff ff       	callq  4014a0 <__cxa_begin_catch@plt>
  4043c8:	48 83 7c 24 18 00    	cmpq   $0x0,0x18(%rsp)
  4043ce:	74 69                	je     404439 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x989>
  4043d0:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
  4043d5:	e8 b6 d1 ff ff       	callq  401590 <_ZdlPv@plt>
  4043da:	e8 71 d2 ff ff       	callq  401650 <__cxa_rethrow@plt>
  4043df:	48 89 c3             	mov    %rax,%rbx
  4043e2:	c5 f8 77             	vzeroupper 
  4043e5:	e8 46 d2 ff ff       	callq  401630 <__cxa_end_catch@plt>
  4043ea:	48 89 df             	mov    %rbx,%rdi
  4043ed:	e8 ae d0 ff ff       	callq  4014a0 <__cxa_begin_catch@plt>
  4043f2:	48 83 7c 24 28 00    	cmpq   $0x0,0x28(%rsp)
  4043f8:	74 3f                	je     404439 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x989>
  4043fa:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  4043ff:	48 39 44 24 28       	cmp    %rax,0x28(%rsp)
  404404:	74 ca                	je     4043d0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x920>
  404406:	48 89 c3             	mov    %rax,%rbx
  404409:	48 89 df             	mov    %rbx,%rdi
  40440c:	e8 3f f5 ff ff       	callq  403950 <_ZN2cv3MatD1Ev>
  404411:	48 83 c3 60          	add    $0x60,%rbx
  404415:	48 39 5c 24 28       	cmp    %rbx,0x28(%rsp)
  40441a:	75 ed                	jne    404409 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x959>
  40441c:	48 83 7c 24 18 00    	cmpq   $0x0,0x18(%rsp)
  404422:	75 ac                	jne    4043d0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x920>
  404424:	eb b4                	jmp    4043da <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x92a>
  404426:	48 89 c3             	mov    %rax,%rbx
  404429:	c5 f8 77             	vzeroupper 
  40442c:	e8 ff d1 ff ff       	callq  401630 <__cxa_end_catch@plt>
  404431:	48 89 df             	mov    %rbx,%rdi
  404434:	e8 67 d0 ff ff       	callq  4014a0 <__cxa_begin_catch@plt>
  404439:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  40443e:	e8 0d f5 ff ff       	callq  403950 <_ZN2cv3MatD1Ev>
  404443:	eb d7                	jmp    40441c <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x96c>
  404445:	48 89 c3             	mov    %rax,%rbx
  404448:	c5 f8 77             	vzeroupper 
  40444b:	e8 e0 d1 ff ff       	callq  401630 <__cxa_end_catch@plt>
  404450:	48 89 df             	mov    %rbx,%rdi
  404453:	e8 88 d1 ff ff       	callq  4015e0 <_Unwind_Resume@plt>
  404458:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40445f:	00 

0000000000404460 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag>:
  404460:	55                   	push   %rbp
  404461:	48 89 e5             	mov    %rsp,%rbp
  404464:	41 56                	push   %r14
  404466:	41 55                	push   %r13
  404468:	41 54                	push   %r12
  40446a:	49 89 d4             	mov    %rdx,%r12
  40446d:	49 29 f4             	sub    %rsi,%r12
  404470:	53                   	push   %rbx
  404471:	48 89 fb             	mov    %rdi,%rbx
  404474:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  404478:	48 83 ec 10          	sub    $0x10,%rsp
  40447c:	48 8b 07             	mov    (%rdi),%rax
  40447f:	48 8b 4f 10          	mov    0x10(%rdi),%rcx
  404483:	48 29 c1             	sub    %rax,%rcx
  404486:	48 c1 f9 02          	sar    $0x2,%rcx
  40448a:	49 39 cc             	cmp    %rcx,%r12
  40448d:	0f 87 0d 04 00 00    	ja     4048a0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x440>
  404493:	4c 8b 4f 08          	mov    0x8(%rdi),%r9
  404497:	4c 89 cf             	mov    %r9,%rdi
  40449a:	48 29 c7             	sub    %rax,%rdi
  40449d:	48 c1 ff 02          	sar    $0x2,%rdi
  4044a1:	49 39 fc             	cmp    %rdi,%r12
  4044a4:	0f 87 2e 01 00 00    	ja     4045d8 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x178>
  4044aa:	4d 85 e4             	test   %r12,%r12
  4044ad:	0f 8e a5 05 00 00    	jle    404a58 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5f8>
  4044b3:	4a 8d 3c a0          	lea    (%rax,%r12,4),%rdi
  4044b7:	4a 8d 0c 26          	lea    (%rsi,%r12,1),%rcx
  4044bb:	48 39 fe             	cmp    %rdi,%rsi
  4044be:	41 0f 93 c0          	setae  %r8b
  4044c2:	48 39 c8             	cmp    %rcx,%rax
  4044c5:	0f 93 c2             	setae  %dl
  4044c8:	41 08 d0             	or     %dl,%r8b
  4044cb:	0f 84 8f 03 00 00    	je     404860 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x400>
  4044d1:	49 83 fc 1f          	cmp    $0x1f,%r12
  4044d5:	0f 86 85 03 00 00    	jbe    404860 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x400>
  4044db:	4d 89 e0             	mov    %r12,%r8
  4044de:	48 8d 50 20          	lea    0x20(%rax),%rdx
  4044e2:	45 31 c9             	xor    %r9d,%r9d
  4044e5:	49 c1 e8 05          	shr    $0x5,%r8
  4044e9:	4c 89 c1             	mov    %r8,%rcx
  4044ec:	48 c1 e1 05          	shl    $0x5,%rcx
  4044f0:	4d 89 ca             	mov    %r9,%r10
  4044f3:	49 83 c1 01          	add    $0x1,%r9
  4044f7:	48 83 ea 80          	sub    $0xffffffffffffff80,%rdx
  4044fb:	49 c1 e2 05          	shl    $0x5,%r10
  4044ff:	c4 a1 7a 6f 04 16    	vmovdqu (%rsi,%r10,1),%xmm0
  404505:	c4 a3 7d 38 44 16 10 	vinserti128 $0x1,0x10(%rsi,%r10,1),%ymm0,%ymm0
  40450c:	01 
  40450d:	c4 e2 7d 30 c8       	vpmovzxbw %xmm0,%ymm1
  404512:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  404518:	c4 e2 7d 23 d1       	vpmovsxwd %xmm1,%ymm2
  40451d:	c4 e3 7d 39 c9 01    	vextracti128 $0x1,%ymm1,%xmm1
  404523:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  404528:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  40452c:	c5 f8 11 92 60 ff ff 	vmovups %xmm2,-0xa0(%rdx)
  404533:	ff 
  404534:	c4 e2 7d 23 c9       	vpmovsxwd %xmm1,%ymm1
  404539:	c4 e3 7d 19 92 70 ff 	vextractf128 $0x1,%ymm2,-0x90(%rdx)
  404540:	ff ff 01 
  404543:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  404547:	c5 f8 11 4a 80       	vmovups %xmm1,-0x80(%rdx)
  40454c:	c4 e3 7d 19 4a 90 01 	vextractf128 $0x1,%ymm1,-0x70(%rdx)
  404553:	c4 e2 7d 23 c8       	vpmovsxwd %xmm0,%ymm1
  404558:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  40455e:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  404562:	c5 f8 11 4a a0       	vmovups %xmm1,-0x60(%rdx)
  404567:	c4 e3 7d 19 4a b0 01 	vextractf128 $0x1,%ymm1,-0x50(%rdx)
  40456e:	c4 e2 7d 23 c0       	vpmovsxwd %xmm0,%ymm0
  404573:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  404577:	c5 f8 11 42 c0       	vmovups %xmm0,-0x40(%rdx)
  40457c:	c4 e3 7d 19 42 d0 01 	vextractf128 $0x1,%ymm0,-0x30(%rdx)
  404583:	4d 39 c1             	cmp    %r8,%r9
  404586:	0f 82 64 ff ff ff    	jb     4044f0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x90>
  40458c:	48 8d 14 88          	lea    (%rax,%rcx,4),%rdx
  404590:	4c 89 e0             	mov    %r12,%rax
  404593:	48 01 ce             	add    %rcx,%rsi
  404596:	48 29 c8             	sub    %rcx,%rax
  404599:	49 39 cc             	cmp    %rcx,%r12
  40459c:	74 1f                	je     4045bd <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x15d>
  40459e:	66 90                	xchg   %ax,%ax
  4045a0:	0f b6 0e             	movzbl (%rsi),%ecx
  4045a3:	48 83 e8 01          	sub    $0x1,%rax
  4045a7:	48 83 c6 01          	add    $0x1,%rsi
  4045ab:	48 83 c2 04          	add    $0x4,%rdx
  4045af:	c5 fa 2a c1          	vcvtsi2ss %ecx,%xmm0,%xmm0
  4045b3:	c5 fa 11 42 fc       	vmovss %xmm0,-0x4(%rdx)
  4045b8:	48 85 c0             	test   %rax,%rax
  4045bb:	7f e3                	jg     4045a0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x140>
  4045bd:	c5 f8 77             	vzeroupper 
  4045c0:	48 89 7b 08          	mov    %rdi,0x8(%rbx)
  4045c4:	48 8d 65 e0          	lea    -0x20(%rbp),%rsp
  4045c8:	5b                   	pop    %rbx
  4045c9:	41 5c                	pop    %r12
  4045cb:	41 5d                	pop    %r13
  4045cd:	41 5e                	pop    %r14
  4045cf:	5d                   	pop    %rbp
  4045d0:	c3                   	retq   
  4045d1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4045d8:	48 01 f7             	add    %rsi,%rdi
  4045db:	49 89 fa             	mov    %rdi,%r10
  4045de:	49 29 f2             	sub    %rsi,%r10
  4045e1:	4d 85 d2             	test   %r10,%r10
  4045e4:	0f 8e 14 01 00 00    	jle    4046fe <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x29e>
  4045ea:	4a 8d 0c 90          	lea    (%rax,%r10,4),%rcx
  4045ee:	4e 8d 04 16          	lea    (%rsi,%r10,1),%r8
  4045f2:	48 39 ce             	cmp    %rcx,%rsi
  4045f5:	41 0f 93 c3          	setae  %r11b
  4045f9:	4c 39 c0             	cmp    %r8,%rax
  4045fc:	0f 93 c1             	setae  %cl
  4045ff:	41 08 cb             	or     %cl,%r11b
  404602:	0f 84 30 02 00 00    	je     404838 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3d8>
  404608:	49 83 fa 1f          	cmp    $0x1f,%r10
  40460c:	0f 86 26 02 00 00    	jbe    404838 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3d8>
  404612:	4d 89 d0             	mov    %r10,%r8
  404615:	48 8d 48 20          	lea    0x20(%rax),%rcx
  404619:	45 31 e4             	xor    %r12d,%r12d
  40461c:	49 c1 e8 05          	shr    $0x5,%r8
  404620:	4d 89 c3             	mov    %r8,%r11
  404623:	49 c1 e3 05          	shl    $0x5,%r11
  404627:	4d 89 e5             	mov    %r12,%r13
  40462a:	49 83 c4 01          	add    $0x1,%r12
  40462e:	48 83 e9 80          	sub    $0xffffffffffffff80,%rcx
  404632:	49 c1 e5 05          	shl    $0x5,%r13
  404636:	c4 a1 7a 6f 04 2e    	vmovdqu (%rsi,%r13,1),%xmm0
  40463c:	c4 a3 7d 38 44 2e 10 	vinserti128 $0x1,0x10(%rsi,%r13,1),%ymm0,%ymm0
  404643:	01 
  404644:	c4 e2 7d 30 c8       	vpmovzxbw %xmm0,%ymm1
  404649:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  40464f:	c4 e2 7d 23 d1       	vpmovsxwd %xmm1,%ymm2
  404654:	c4 e3 7d 39 c9 01    	vextracti128 $0x1,%ymm1,%xmm1
  40465a:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  40465f:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  404663:	c5 f8 11 91 60 ff ff 	vmovups %xmm2,-0xa0(%rcx)
  40466a:	ff 
  40466b:	c4 e2 7d 23 c9       	vpmovsxwd %xmm1,%ymm1
  404670:	c4 e3 7d 19 91 70 ff 	vextractf128 $0x1,%ymm2,-0x90(%rcx)
  404677:	ff ff 01 
  40467a:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  40467e:	c5 f8 11 49 80       	vmovups %xmm1,-0x80(%rcx)
  404683:	c4 e3 7d 19 49 90 01 	vextractf128 $0x1,%ymm1,-0x70(%rcx)
  40468a:	c4 e2 7d 23 c8       	vpmovsxwd %xmm0,%ymm1
  40468f:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  404695:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  404699:	c5 f8 11 49 a0       	vmovups %xmm1,-0x60(%rcx)
  40469e:	c4 e3 7d 19 49 b0 01 	vextractf128 $0x1,%ymm1,-0x50(%rcx)
  4046a5:	c4 e2 7d 23 c0       	vpmovsxwd %xmm0,%ymm0
  4046aa:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  4046ae:	c5 f8 11 41 c0       	vmovups %xmm0,-0x40(%rcx)
  4046b3:	c4 e3 7d 19 41 d0 01 	vextractf128 $0x1,%ymm0,-0x30(%rcx)
  4046ba:	4d 39 c4             	cmp    %r8,%r12
  4046bd:	0f 82 64 ff ff ff    	jb     404627 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x1c7>
  4046c3:	4a 8d 0c 98          	lea    (%rax,%r11,4),%rcx
  4046c7:	4c 89 d0             	mov    %r10,%rax
  4046ca:	4e 8d 04 1e          	lea    (%rsi,%r11,1),%r8
  4046ce:	4c 29 d8             	sub    %r11,%rax
  4046d1:	4d 39 da             	cmp    %r11,%r10
  4046d4:	74 28                	je     4046fe <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x29e>
  4046d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4046dd:	00 00 00 
  4046e0:	41 0f b6 30          	movzbl (%r8),%esi
  4046e4:	48 83 e8 01          	sub    $0x1,%rax
  4046e8:	49 83 c0 01          	add    $0x1,%r8
  4046ec:	48 83 c1 04          	add    $0x4,%rcx
  4046f0:	c5 fa 2a c6          	vcvtsi2ss %esi,%xmm0,%xmm0
  4046f4:	c5 fa 11 41 fc       	vmovss %xmm0,-0x4(%rcx)
  4046f9:	48 85 c0             	test   %rax,%rax
  4046fc:	7f e2                	jg     4046e0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x280>
  4046fe:	48 29 fa             	sub    %rdi,%rdx
  404701:	4d 89 c8             	mov    %r9,%r8
  404704:	48 85 d2             	test   %rdx,%rdx
  404707:	0f 8e 10 01 00 00    	jle    40481d <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3bd>
  40470d:	4d 8d 04 91          	lea    (%r9,%rdx,4),%r8
  404711:	48 8d 0c 17          	lea    (%rdi,%rdx,1),%rcx
  404715:	4c 39 c7             	cmp    %r8,%rdi
  404718:	40 0f 93 c6          	setae  %sil
  40471c:	49 39 c9             	cmp    %rcx,%r9
  40471f:	0f 93 c0             	setae  %al
  404722:	40 08 c6             	or     %al,%sil
  404725:	0f 84 55 01 00 00    	je     404880 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x420>
  40472b:	48 83 fa 1f          	cmp    $0x1f,%rdx
  40472f:	0f 86 4b 01 00 00    	jbe    404880 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x420>
  404735:	48 89 d1             	mov    %rdx,%rcx
  404738:	49 8d 41 20          	lea    0x20(%r9),%rax
  40473c:	31 f6                	xor    %esi,%esi
  40473e:	48 c1 e9 05          	shr    $0x5,%rcx
  404742:	49 89 ca             	mov    %rcx,%r10
  404745:	49 c1 e2 05          	shl    $0x5,%r10
  404749:	49 89 f3             	mov    %rsi,%r11
  40474c:	48 83 c6 01          	add    $0x1,%rsi
  404750:	48 83 e8 80          	sub    $0xffffffffffffff80,%rax
  404754:	49 c1 e3 05          	shl    $0x5,%r11
  404758:	c4 a1 7a 6f 04 1f    	vmovdqu (%rdi,%r11,1),%xmm0
  40475e:	c4 a3 7d 38 44 1f 10 	vinserti128 $0x1,0x10(%rdi,%r11,1),%ymm0,%ymm0
  404765:	01 
  404766:	c4 e2 7d 30 c8       	vpmovzxbw %xmm0,%ymm1
  40476b:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  404771:	c4 e2 7d 23 d1       	vpmovsxwd %xmm1,%ymm2
  404776:	c4 e3 7d 39 c9 01    	vextracti128 $0x1,%ymm1,%xmm1
  40477c:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  404781:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  404785:	c5 f8 11 90 60 ff ff 	vmovups %xmm2,-0xa0(%rax)
  40478c:	ff 
  40478d:	c4 e2 7d 23 c9       	vpmovsxwd %xmm1,%ymm1
  404792:	c4 e3 7d 19 90 70 ff 	vextractf128 $0x1,%ymm2,-0x90(%rax)
  404799:	ff ff 01 
  40479c:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  4047a0:	c5 f8 11 48 80       	vmovups %xmm1,-0x80(%rax)
  4047a5:	c4 e3 7d 19 48 90 01 	vextractf128 $0x1,%ymm1,-0x70(%rax)
  4047ac:	c4 e2 7d 23 c8       	vpmovsxwd %xmm0,%ymm1
  4047b1:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  4047b7:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  4047bb:	c5 f8 11 48 a0       	vmovups %xmm1,-0x60(%rax)
  4047c0:	c4 e3 7d 19 48 b0 01 	vextractf128 $0x1,%ymm1,-0x50(%rax)
  4047c7:	c4 e2 7d 23 c0       	vpmovsxwd %xmm0,%ymm0
  4047cc:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  4047d0:	c5 f8 11 40 c0       	vmovups %xmm0,-0x40(%rax)
  4047d5:	c4 e3 7d 19 40 d0 01 	vextractf128 $0x1,%ymm0,-0x30(%rax)
  4047dc:	48 39 ce             	cmp    %rcx,%rsi
  4047df:	0f 82 64 ff ff ff    	jb     404749 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x2e9>
  4047e5:	48 89 d0             	mov    %rdx,%rax
  4047e8:	4a 8d 34 17          	lea    (%rdi,%r10,1),%rsi
  4047ec:	4b 8d 0c 91          	lea    (%r9,%r10,4),%rcx
  4047f0:	4c 29 d0             	sub    %r10,%rax
  4047f3:	4c 39 d2             	cmp    %r10,%rdx
  4047f6:	74 25                	je     40481d <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3bd>
  4047f8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4047ff:	00 
  404800:	0f b6 16             	movzbl (%rsi),%edx
  404803:	48 83 e8 01          	sub    $0x1,%rax
  404807:	48 83 c6 01          	add    $0x1,%rsi
  40480b:	48 83 c1 04          	add    $0x4,%rcx
  40480f:	c5 fa 2a c2          	vcvtsi2ss %edx,%xmm0,%xmm0
  404813:	c5 fa 11 41 fc       	vmovss %xmm0,-0x4(%rcx)
  404818:	48 85 c0             	test   %rax,%rax
  40481b:	7f e3                	jg     404800 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3a0>
  40481d:	4c 89 43 08          	mov    %r8,0x8(%rbx)
  404821:	c5 f8 77             	vzeroupper 
  404824:	48 8d 65 e0          	lea    -0x20(%rbp),%rsp
  404828:	5b                   	pop    %rbx
  404829:	41 5c                	pop    %r12
  40482b:	41 5d                	pop    %r13
  40482d:	41 5e                	pop    %r14
  40482f:	5d                   	pop    %rbp
  404830:	c3                   	retq   
  404831:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  404838:	0f b6 0e             	movzbl (%rsi),%ecx
  40483b:	48 83 c6 01          	add    $0x1,%rsi
  40483f:	48 83 c0 04          	add    $0x4,%rax
  404843:	c5 fa 2a c1          	vcvtsi2ss %ecx,%xmm0,%xmm0
  404847:	c5 fa 11 40 fc       	vmovss %xmm0,-0x4(%rax)
  40484c:	49 39 f0             	cmp    %rsi,%r8
  40484f:	75 e7                	jne    404838 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3d8>
  404851:	e9 a8 fe ff ff       	jmpq   4046fe <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x29e>
  404856:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40485d:	00 00 00 
  404860:	0f b6 16             	movzbl (%rsi),%edx
  404863:	48 83 c6 01          	add    $0x1,%rsi
  404867:	48 83 c0 04          	add    $0x4,%rax
  40486b:	c5 fa 2a c2          	vcvtsi2ss %edx,%xmm0,%xmm0
  40486f:	c5 fa 11 40 fc       	vmovss %xmm0,-0x4(%rax)
  404874:	48 39 f1             	cmp    %rsi,%rcx
  404877:	75 e7                	jne    404860 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x400>
  404879:	e9 42 fd ff ff       	jmpq   4045c0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x160>
  40487e:	66 90                	xchg   %ax,%ax
  404880:	0f b6 07             	movzbl (%rdi),%eax
  404883:	48 83 c7 01          	add    $0x1,%rdi
  404887:	49 83 c1 04          	add    $0x4,%r9
  40488b:	c5 fa 2a c0          	vcvtsi2ss %eax,%xmm0,%xmm0
  40488f:	c4 c1 7a 11 41 fc    	vmovss %xmm0,-0x4(%r9)
  404895:	48 39 f9             	cmp    %rdi,%rcx
  404898:	75 e6                	jne    404880 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x420>
  40489a:	eb 81                	jmp    40481d <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3bd>
  40489c:	0f 1f 40 00          	nopl   0x0(%rax)
  4048a0:	4d 85 e4             	test   %r12,%r12
  4048a3:	0f 84 9f 01 00 00    	je     404a48 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5e8>
  4048a9:	48 b8 ff ff ff ff ff 	movabs $0x3fffffffffffffff,%rax
  4048b0:	ff ff 3f 
  4048b3:	49 39 c4             	cmp    %rax,%r12
  4048b6:	0f 87 a4 01 00 00    	ja     404a60 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x600>
  4048bc:	4e 8d 34 a5 00 00 00 	lea    0x0(,%r12,4),%r14
  4048c3:	00 
  4048c4:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  4048c9:	4c 89 f7             	mov    %r14,%rdi
  4048cc:	e8 0f cc ff ff       	callq  4014e0 <_Znwm@plt>
  4048d1:	4d 85 e4             	test   %r12,%r12
  4048d4:	49 89 c5             	mov    %rax,%r13
  4048d7:	48 8b 3b             	mov    (%rbx),%rdi
  4048da:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  4048df:	0f 8e 0d 01 00 00    	jle    4049f2 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x592>
  4048e5:	4a 8d 04 30          	lea    (%rax,%r14,1),%rax
  4048e9:	4e 8d 04 26          	lea    (%rsi,%r12,1),%r8
  4048ed:	48 39 c6             	cmp    %rax,%rsi
  4048f0:	0f 93 c2             	setae  %dl
  4048f3:	4d 39 e8             	cmp    %r13,%r8
  4048f6:	0f 96 c0             	setbe  %al
  4048f9:	08 c2                	or     %al,%dl
  4048fb:	0f 84 1f 01 00 00    	je     404a20 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5c0>
  404901:	49 83 fc 1f          	cmp    $0x1f,%r12
  404905:	0f 86 15 01 00 00    	jbe    404a20 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5c0>
  40490b:	4c 89 e1             	mov    %r12,%rcx
  40490e:	49 8d 55 20          	lea    0x20(%r13),%rdx
  404912:	45 31 c0             	xor    %r8d,%r8d
  404915:	48 c1 e9 05          	shr    $0x5,%rcx
  404919:	48 89 c8             	mov    %rcx,%rax
  40491c:	48 c1 e0 05          	shl    $0x5,%rax
  404920:	4d 89 c1             	mov    %r8,%r9
  404923:	49 83 c0 01          	add    $0x1,%r8
  404927:	48 83 ea 80          	sub    $0xffffffffffffff80,%rdx
  40492b:	49 c1 e1 05          	shl    $0x5,%r9
  40492f:	c4 a1 7a 6f 04 0e    	vmovdqu (%rsi,%r9,1),%xmm0
  404935:	c4 a3 7d 38 44 0e 10 	vinserti128 $0x1,0x10(%rsi,%r9,1),%ymm0,%ymm0
  40493c:	01 
  40493d:	c4 e2 7d 30 c8       	vpmovzxbw %xmm0,%ymm1
  404942:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  404948:	c4 e2 7d 23 d1       	vpmovsxwd %xmm1,%ymm2
  40494d:	c4 e3 7d 39 c9 01    	vextracti128 $0x1,%ymm1,%xmm1
  404953:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  404958:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  40495c:	c5 f8 11 92 60 ff ff 	vmovups %xmm2,-0xa0(%rdx)
  404963:	ff 
  404964:	c4 e2 7d 23 c9       	vpmovsxwd %xmm1,%ymm1
  404969:	c4 e3 7d 19 92 70 ff 	vextractf128 $0x1,%ymm2,-0x90(%rdx)
  404970:	ff ff 01 
  404973:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  404977:	c5 f8 11 4a 80       	vmovups %xmm1,-0x80(%rdx)
  40497c:	c4 e3 7d 19 4a 90 01 	vextractf128 $0x1,%ymm1,-0x70(%rdx)
  404983:	c4 e2 7d 23 c8       	vpmovsxwd %xmm0,%ymm1
  404988:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  40498e:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  404992:	c5 f8 11 4a a0       	vmovups %xmm1,-0x60(%rdx)
  404997:	c4 e3 7d 19 4a b0 01 	vextractf128 $0x1,%ymm1,-0x50(%rdx)
  40499e:	c4 e2 7d 23 c0       	vpmovsxwd %xmm0,%ymm0
  4049a3:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  4049a7:	c5 f8 11 42 c0       	vmovups %xmm0,-0x40(%rdx)
  4049ac:	c4 e3 7d 19 42 d0 01 	vextractf128 $0x1,%ymm0,-0x30(%rdx)
  4049b3:	4c 39 c1             	cmp    %r8,%rcx
  4049b6:	0f 87 64 ff ff ff    	ja     404920 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x4c0>
  4049bc:	4c 89 e2             	mov    %r12,%rdx
  4049bf:	48 01 c6             	add    %rax,%rsi
  4049c2:	49 8d 4c 85 00       	lea    0x0(%r13,%rax,4),%rcx
  4049c7:	48 29 c2             	sub    %rax,%rdx
  4049ca:	49 39 c4             	cmp    %rax,%r12
  4049cd:	74 20                	je     4049ef <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x58f>
  4049cf:	90                   	nop
  4049d0:	44 0f b6 06          	movzbl (%rsi),%r8d
  4049d4:	48 83 ea 01          	sub    $0x1,%rdx
  4049d8:	48 83 c6 01          	add    $0x1,%rsi
  4049dc:	48 83 c1 04          	add    $0x4,%rcx
  4049e0:	c4 c1 7a 2a c0       	vcvtsi2ss %r8d,%xmm0,%xmm0
  4049e5:	c5 fa 11 41 fc       	vmovss %xmm0,-0x4(%rcx)
  4049ea:	48 85 d2             	test   %rdx,%rdx
  4049ed:	7f e1                	jg     4049d0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x570>
  4049ef:	c5 f8 77             	vzeroupper 
  4049f2:	48 85 ff             	test   %rdi,%rdi
  4049f5:	74 05                	je     4049fc <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x59c>
  4049f7:	e8 94 cb ff ff       	callq  401590 <_ZdlPv@plt>
  4049fc:	4c 89 2b             	mov    %r13,(%rbx)
  4049ff:	4d 01 f5             	add    %r14,%r13
  404a02:	4c 89 6b 08          	mov    %r13,0x8(%rbx)
  404a06:	4c 89 6b 10          	mov    %r13,0x10(%rbx)
  404a0a:	48 8d 65 e0          	lea    -0x20(%rbp),%rsp
  404a0e:	5b                   	pop    %rbx
  404a0f:	41 5c                	pop    %r12
  404a11:	41 5d                	pop    %r13
  404a13:	41 5e                	pop    %r14
  404a15:	5d                   	pop    %rbp
  404a16:	c3                   	retq   
  404a17:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  404a1e:	00 00 
  404a20:	4c 89 ea             	mov    %r13,%rdx
  404a23:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  404a28:	0f b6 0e             	movzbl (%rsi),%ecx
  404a2b:	48 83 c6 01          	add    $0x1,%rsi
  404a2f:	48 83 c2 04          	add    $0x4,%rdx
  404a33:	c5 fa 2a c1          	vcvtsi2ss %ecx,%xmm0,%xmm0
  404a37:	c5 fa 11 42 fc       	vmovss %xmm0,-0x4(%rdx)
  404a3c:	49 39 f0             	cmp    %rsi,%r8
  404a3f:	75 e7                	jne    404a28 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5c8>
  404a41:	eb af                	jmp    4049f2 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x592>
  404a43:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  404a48:	48 89 c7             	mov    %rax,%rdi
  404a4b:	45 31 f6             	xor    %r14d,%r14d
  404a4e:	45 31 ed             	xor    %r13d,%r13d
  404a51:	eb 9f                	jmp    4049f2 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x592>
  404a53:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  404a58:	48 89 c7             	mov    %rax,%rdi
  404a5b:	e9 60 fb ff ff       	jmpq   4045c0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x160>
  404a60:	e8 6b ca ff ff       	callq  4014d0 <_ZSt17__throw_bad_allocv@plt>
  404a65:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  404a6c:	00 00 00 
  404a6f:	90                   	nop

0000000000404a70 <__libc_csu_init>:
  404a70:	41 57                	push   %r15
  404a72:	41 89 ff             	mov    %edi,%r15d
  404a75:	41 56                	push   %r14
  404a77:	49 89 f6             	mov    %rsi,%r14
  404a7a:	41 55                	push   %r13
  404a7c:	49 89 d5             	mov    %rdx,%r13
  404a7f:	41 54                	push   %r12
  404a81:	4c 8d 25 30 12 20 00 	lea    0x201230(%rip),%r12        # 605cb8 <__frame_dummy_init_array_entry>
  404a88:	55                   	push   %rbp
  404a89:	48 8d 2d 38 12 20 00 	lea    0x201238(%rip),%rbp        # 605cc8 <__init_array_end>
  404a90:	53                   	push   %rbx
  404a91:	4c 29 e5             	sub    %r12,%rbp
  404a94:	31 db                	xor    %ebx,%ebx
  404a96:	48 c1 fd 03          	sar    $0x3,%rbp
  404a9a:	48 83 ec 08          	sub    $0x8,%rsp
  404a9e:	e8 cd c9 ff ff       	callq  401470 <_init>
  404aa3:	48 85 ed             	test   %rbp,%rbp
  404aa6:	74 1e                	je     404ac6 <__libc_csu_init+0x56>
  404aa8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  404aaf:	00 
  404ab0:	4c 89 ea             	mov    %r13,%rdx
  404ab3:	4c 89 f6             	mov    %r14,%rsi
  404ab6:	44 89 ff             	mov    %r15d,%edi
  404ab9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  404abd:	48 83 c3 01          	add    $0x1,%rbx
  404ac1:	48 39 eb             	cmp    %rbp,%rbx
  404ac4:	75 ea                	jne    404ab0 <__libc_csu_init+0x40>
  404ac6:	48 83 c4 08          	add    $0x8,%rsp
  404aca:	5b                   	pop    %rbx
  404acb:	5d                   	pop    %rbp
  404acc:	41 5c                	pop    %r12
  404ace:	41 5d                	pop    %r13
  404ad0:	41 5e                	pop    %r14
  404ad2:	41 5f                	pop    %r15
  404ad4:	c3                   	retq   
  404ad5:	90                   	nop
  404ad6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  404add:	00 00 00 

0000000000404ae0 <__libc_csu_fini>:
  404ae0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000404ae4 <_fini>:
  404ae4:	48 83 ec 08          	sub    $0x8,%rsp
  404ae8:	48 83 c4 08          	add    $0x8,%rsp
  404aec:	c3                   	retq   
